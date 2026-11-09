"""The small OpenAI-specific boundary: Responses, streaming, and Batch."""

from openai import OpenAI


TERMINAL_RESPONSES = {'completed', 'incomplete', 'failed', 'cancelled'}
TERMINAL_BATCHES = {'completed', 'failed', 'expired', 'cancelled'}


class OpenAIProvider:
    def __init__(self, client=None, base_url=None):
        # Disable invisible SDK retries so interrupted attempts remain explicit.
        self.client = client if client is not None else OpenAI(base_url=base_url, max_retries=0)

    def generate(self, body, stream, on_event):
        if not stream:
            response = self.client.responses.create(**body).model_dump(mode='json')
            on_event({'type': 'response.' + response['status'], 'response': response})
            return response
        response = None
        with self.client.responses.create(**body, stream=True) as events:
            for event in events:
                data = event.model_dump(mode='json')
                on_event(data)
                if data['type'] in ('response.completed', 'response.incomplete', 'response.failed'):
                    response = data['response']
                elif data['type'] == 'error':
                    raise RuntimeError(data.get('message', 'OpenAI stream error.'))
        if response is None:
            raise RuntimeError('The stream ended without a final response.')
        return response

    def retrieve_response(self, response_id):
        return self.client.responses.retrieve(response_id).model_dump(mode='json')

    def submit_batch(self, path, submission_id):
        with path.open('rb') as handle:
            uploaded = self.client.files.create(file=handle, purpose='batch')
        return self.client.batches.create(
            input_file_id=uploaded.id, endpoint='/v1/responses',
            completion_window='24h', metadata={'submission_id': submission_id},
        ).model_dump(mode='json')

    def find_batch(self, submission_id):
        for batch in self.client.batches.list(limit=100):
            if (batch.metadata or {}).get('submission_id') == submission_id:
                return batch.model_dump(mode='json')
        return None

    def retrieve_batch(self, batch_id):
        return self.client.batches.retrieve(batch_id).model_dump(mode='json')

    def download(self, file_id):
        return self.client.files.content(file_id).text


def format_response(sample_id, body, attempt_id):
    text = []
    refusals = []
    for item in body.get('output') or []:
        if item.get('type') == 'message' and item.get('role', 'assistant') == 'assistant':
            for part in item.get('content') or []:
                if part.get('type') == 'output_text':
                    text.append(part['text'])
                elif part.get('type') == 'refusal':
                    refusals.append(part['refusal'])
    return {
        'sample_id': sample_id, 'attempt_id': attempt_id,
        'status': 'refused' if refusals else body.get('status', 'failed'),
        'text': ''.join(text), 'response_id': body.get('id'),
        'model': body.get('model'), 'usage': body.get('usage'),
        'error': body.get('error'), 'refusal': '\n'.join(refusals) or None,
        'incomplete_details': body.get('incomplete_details'),
    }
