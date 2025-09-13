# %%
import os
import re

import numpy as np
import ollama
import torch
import wikipedia
from bark.api import generate_audio

# Bark TTS 라이브러리를 import 합니다.
from bark.generation import preload_models
from scipy.io.wavfile import write as write_wav  # Bark 오디오 저장을 위해 추가


# --- 텍스트 정제, 위키백과, Ollama 프롬프트 함수는 이전과 동일합니다 ---
def clean_text_for_tts(text):
    print("🧹 불필요한 기호를 제거하여 대본을 정제합니다...")
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[\*_]{1,2}", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Bark는 영어 외 언어에서 줄바꿈이 있으면 불안정할 수 있으므로 제거합니다.
    text = text.replace("\n", " ").replace("\r", " ")
    print("✅ 대본 정제 완료.")
    return text


def get_wikipedia_summary(keyword, lang="ko", max_chars=1500):
    print(f"🌐 '{keyword}'에 대한 위키백과 요약문을 가져옵니다...")
    try:
        wikipedia.set_lang(lang)
        page = wikipedia.page(keyword, auto_suggest=False)
        summary = page.content.split("==")[0].strip()
        print(f"✅ 위키백과 요약문 확보 완료.")
        return summary[:max_chars]
    except Exception as e:
        print(f"🚨 위키백과 정보를 가져오는 중 오류 발생: {e}")
        return None


def create_youtube_script_prompt(summary, topic, duration_minutes=10):
    target_length = duration_minutes * 250
    prompt = f"""
    당신은 해박한 지식을 가진 '유튜브 역사 스토리텔러'입니다.
    주어진 주제와 요약문을 바탕으로, 청중이 흥미를 느끼고 몰입할 수 있는 **{duration_minutes}분 분량(약 {target_length}자 내외)의 유튜브 영상 대본**을 작성해주세요.
    ### 주제: {topic}
    ### 위키백과 기반 핵심 요약문: {summary}
    ### 대본 작성 지침:
    1.  **도입부:** 청중의 호기심을 자극하는 질문이나 흥미로운 사실로 시작해주세요.
    2.  **본문:** 요약문 내용을 바탕으로 당신의 지식을 더해 이야기를 풍부하게 만들어주세요.
    3.  **마무리:** 내용을 깔끔하게 요약하고, 청중에게 여운을 남기는 메시지나 다음 영상에 대한 기대감을 주는 말로 끝내주세요.
    4.  **어조:** 딱딱한 설명이 아닌, 친구에게 재미있는 옛날이야기를 들려주듯 친근하고 이해하기 쉬운 말투를 사용해주세요.
    5.  **출력 형식:** 오직 말로 읽을 대본만 작성해주세요.
    """
    return prompt.strip()


# --- 메인 로직 ---
def main():
    # --- 설정 변수 ---
    KEYWORD = "장영실"
    OLLAMA_MODEL = "gpt-oss:latest"
    OUTPUT_SCRIPT_FILE = f"{KEYWORD}_script.txt"
    OUTPUT_AUDIO_FILE = f"{KEYWORD}_audio_bark.wav"  # Bark 출력 파일
    BARK_SAMPLE_RATE = 24_000  # Bark의 기본 샘플링 레이트

    # --- 1. Bark 모델 로딩 ---
    print("🔊 Bark TTS 모델을 로딩합니다...")
    print(
        "   (⚠️ 첫 실행 시, 필요한 모델 파일을 다운로드하므로 몇 분 이상 소요될 수 있습니다.)"
    )
    try:
        preload_models()
        print("✅ Bark 모델 로딩 완료.")
    except Exception as e:
        print(f"🚨 Bark 모델 로딩 중 오류 발생: {e}")
        return

    # --- 2. 대본 생성 (Ollama) ---
    wiki_summary = get_wikipedia_summary(KEYWORD)
    if not wiki_summary:
        return

    print(f"🤖 Ollama({OLLAMA_MODEL})에 대본 생성을 요청합니다...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": create_youtube_script_prompt(wiki_summary, KEYWORD),
                }
            ],
        )
        generated_script = response["message"]["content"]
        cleaned_script = clean_text_for_tts(generated_script)

        with open(OUTPUT_SCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(cleaned_script)
        print(f"💾 대본을 '{OUTPUT_SCRIPT_FILE}' 파일로 저장했습니다.")

        # --- 3. 음성 합성 (Bark) ---
        print(
            f"🎤 Bark로 음성 합성을 시작합니다... (대본 길이에 따라 시간이 걸릴 수 있습니다)"
        )
        # Bark는 아직 한국어 지원이 완벽하지 않아, 효과음을 내거나 [laughs] 같은 부분을 영어로 인식할 수 있습니다.
        # 한국어 목소리 지정을 위해 스크립트 앞에 "[ko]" 를 붙여줄 수 있습니다. (효과는 모델에 따라 다름)
        # text_prompt = "[ko]" + cleaned_script
        audio_array = generate_audio(cleaned_script)

        # --- 4. 오디오 파일 저장 ---
        print(f"💾 생성된 오디오를 '{OUTPUT_AUDIO_FILE}' 파일로 저장합니다...")
        write_wav(OUTPUT_AUDIO_FILE, BARK_SAMPLE_RATE, audio_array)

        print(f"🎉 최종 성공! '{OUTPUT_AUDIO_FILE}' 음성 파일이 생성되었습니다.")

    except Exception as e:
        print(f"🚨 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    main()

# %%
