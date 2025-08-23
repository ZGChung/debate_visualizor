import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_analyze_endpoint():
    payload = {"text": "Alice: Hello\nBob: Hi Alice\nAlice: How are you"}
    r = client.post('/analyze', content=json.dumps(payload))
    assert r.status_code == 200
    data = r.json()
    assert 'frequencies' in data
    assert 'tug_of_war' in data
    assert isinstance(data['top_words'], list)
