import base64
import hmac
import json
import os
import uuid
import urllib.error
import urllib.request


TTS_ENDPOINT = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"


def make_response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json; charset=utf-8"
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


def decode_event(event):
    """
    兼容阿里 FC：
    - 控制台测试时 event 可能是 bytes / str
    - HTTP Trigger 时 event 会是 dict
    """
    if event is None:
        return {}

    if isinstance(event, bytes):
        event = event.decode("utf-8")

    if isinstance(event, str):
        event = json.loads(event)

    if not isinstance(event, dict):
        return {}

    return event


def get_payload(event_obj):
    """
    兼容两种输入：

    1. 控制台直接测试：
       {
         "token": "...",
         "text": "...",
         "instruction": "..."
       }

    2. HTTP Trigger：
       body 中包含 JSON 字符串
    """
    body = event_obj.get("body")

    if body is None:
        return event_obj

    if event_obj.get("isBase64Encoded") and isinstance(body, str):
        body = base64.b64decode(body).decode("utf-8")

    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    if isinstance(body, dict):
        return body

    return {}


def verify_proxy_token(payload):
    expected = os.environ.get("PROXY_TOKEN", "").strip()

    if not expected:
        raise RuntimeError("Missing PROXY_TOKEN")

    supplied = str(payload.get("token", "")).strip()

    return bool(supplied) and hmac.compare_digest(
        supplied,
        expected
    )


def synthesize(text, instruction=""):
    api_key = os.environ.get(
        "DOUBAO_API_KEY",
        ""
    ).strip()

    resource_id = os.environ.get(
        "DOUBAO_RESOURCE_ID",
        "seed-icl-2.0"
    ).strip()

    speaker_id = os.environ.get(
        "DOUBAO_SPEAKER_ID",
        ""
    ).strip()

    if not api_key:
        raise RuntimeError("Missing DOUBAO_API_KEY")

    if not speaker_id:
        raise RuntimeError("Missing DOUBAO_SPEAKER_ID")

    req_params = {
        "text": text,
        "speaker": speaker_id,
        "sample_rate": 24000,
        "audio_params": {
            "format": "mp3",
            "speech_rate": 0,
            "loudness_rate": 0,
            "bit_rate": 64000
        }
    }

    # 声音复刻 2.0 的总体语音指令：
    # V3 协议通过 additions(JSON 字符串) 传 context_texts。
    # 本代理只传当前 block 的 instruction，不传任何历史对话上下文。
    if instruction:
        req_params["additions"] = json.dumps(
            {
                "context_texts": [instruction]
            },
            ensure_ascii=False
        )

    request_body = {
        "user": {
            "uid": "personal_ai_frontend"
        },
        "req_params": req_params
    }

    request = urllib.request.Request(
        TTS_ENDPOINT,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4())
        },
        method="POST"
    )

    audio_chunks = []

    try:
        with urllib.request.urlopen(
            request,
            timeout=55
        ) as response:

            for raw_line in response:
                line = raw_line.decode(
                    "utf-8"
                ).strip()

                if not line.startswith("data:"):
                    continue

                event_text = line[5:].strip()

                if (
                    not event_text
                    or event_text == "[DONE]"
                ):
                    continue

                try:
                    event_data = json.loads(
                        event_text
                    )
                except json.JSONDecodeError:
                    continue

                code = event_data.get("code", 0)

                if code not in (0, 20000000):
                    message = event_data.get(
                        "message",
                        ""
                    )

                    raise RuntimeError(
                        f"Doubao API error "
                        f"{code}: {message}"
                    )

                audio_base64 = event_data.get(
                    "data"
                )

                if audio_base64:
                    audio_chunks.append(
                        base64.b64decode(
                            audio_base64
                        )
                    )

    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode(
                "utf-8"
            )
        except Exception:
            detail = str(error)

        raise RuntimeError(
            f"Doubao HTTP "
            f"{error.code}: {detail}"
        ) from error

    if not audio_chunks:
        raise RuntimeError(
            "Doubao returned no audio data"
        )

    return b"".join(audio_chunks)


def handler(event, context):
    try:
        event_obj = decode_event(event)
        payload = get_payload(event_obj)

        # 先验证我们自己的 Proxy Token。
        # 验证失败时不会调用豆包，不消耗 TTS 额度。
        if not verify_proxy_token(payload):
            return make_response(
                401,
                {
                    "ok": False,
                    "error": "unauthorized"
                }
            )

        text = str(
            payload.get("text", "")
        ).strip()

        instruction = str(
            payload.get("instruction", "")
        ).strip()

        if not text:
            return make_response(
                400,
                {
                    "ok": False,
                    "error": "text is required"
                }
            )

        if len(text) > 3000:
            return make_response(
                400,
                {
                    "ok": False,
                    "error": "text is too long"
                }
            )

        if len(instruction) > 1200:
            return make_response(
                400,
                {
                    "ok": False,
                    "error": "instruction is too long"
                }
            )

        audio_bytes = synthesize(
            text,
            instruction
        )

        return make_response(
            200,
            {
                "ok": True,
                "contentType": "audio/mpeg",
                "audioBase64": base64.b64encode(
                    audio_bytes
                ).decode("ascii"),
                "bytes": len(audio_bytes),
                "instructionApplied": bool(instruction)
            }
        )

    except Exception as error:
        return make_response(
            500,
            {
                "ok": False,
                "error": str(error)
            }
        )
