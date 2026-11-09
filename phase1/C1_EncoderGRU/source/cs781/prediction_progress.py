"""Atomic resumable predictions bound to prompts and complete inference provenance."""
import hashlib
import json
from pathlib import Path

from .io import write_json_atomic
from .submission import load_json_strict, validate_predictions


class PredictionProgress:
    def __init__(self, path, prompts, fingerprint):
        self.path, self.prompts = Path(path), dict(prompts)
        if (not self.prompts or any(not isinstance(k, str) or not k.strip()
                                  or not isinstance(v, str) or len(v.split()) != 512
                                  for k, v in self.prompts.items())):
            raise ValueError('Progress requires exact 512-word prompts with string IDs')
        prompt_hash = hashlib.sha256(json.dumps(self.prompts, sort_keys=True,
                                                ensure_ascii=False).encode()).hexdigest()
        self.provenance = {'inputs': fingerprint, 'prompt_sha256': prompt_hash}
        self.predictions, self.failures = {}, {}
        if self.path.exists():
            record = load_json_strict(self.path)
            if not isinstance(record, dict) or record.get('provenance') != self.provenance:
                raise ValueError('Prediction progress provenance differs; use a new output path')
            self.predictions, self.failures = record.get('predictions'), record.get('failures')
            if (not isinstance(self.predictions, dict) or not isinstance(self.failures, dict)
                    or not set(self.predictions).issubset(self.prompts)
                    or not set(self.failures).issubset(self.prompts)
                    or set(self.predictions) & set(self.failures)):
                raise ValueError('Malformed saved prediction progress')
            for key, text in self.predictions.items():
                validate_predictions({key: text}, {key: self.prompts[key]})
        else:
            self._save()

    def pending(self):
        return (key for key in self.prompts if key not in self.predictions)

    def _save(self):
        write_json_atomic(self.path, {'provenance': self.provenance,
                                     'predictions': self.predictions, 'failures': self.failures})

    def record(self, key, text):
        if key not in self.prompts:
            raise ValueError('Prediction ID is not in declared prompts')
        validate_predictions({key: text}, {key: self.prompts[key]})
        self.predictions[key] = text
        self.failures.pop(key, None)
        self._save()

    def failure(self, key, message):
        if key not in self.prompts or key in self.predictions:
            raise ValueError('Failure ID is unknown or already complete')
        self.failures[key] = str(message)
        self._save()

    def complete(self):
        if len(self.predictions) != len(self.prompts) or self.failures:
            raise ValueError('Generation is incomplete; saved records can be resumed')
        validate_predictions(self.predictions, self.prompts)
        return dict(self.predictions)
