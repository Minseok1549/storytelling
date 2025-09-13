# %%
import os
import re
import time

import numpy as np
import ollama
import torch
import wikipedia
from bark.api import generate_audio
from bark.generation import preload_models
from scipy.io.wavfile import write as write_wav


# --- 텍스트 정제 및 위키백과 함수는 이전과 동일 ---
def clean_text_for_tts(text):
    print("🧹 불필요한 기호를 제거하여 대본을 정제합니다...")
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[\*_]{1,2}", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    print("✅ 대본 정제 완료.")
    return text


def get_wikipedia_summary(
    keyword, lang="ko", max_chars=3000
):  # 정보량 확보를 위해 3000자로 늘림
    print(f"🌐 '{keyword}'에 대한 위키백과 요약문을 가져옵니다 (최대 {max_chars}자)...")
    try:
        wikipedia.set_lang(lang)
        page = wikipedia.page(keyword, auto_suggest=False)
        summary = page.content.split("==")[0].strip()
        print(f"✅ 위키백과 요약문 확보 완료 (길이: {len(summary)}자).")
        return summary[:max_chars]
    except Exception as e:
        print(f"🚨 위키백과 정보를 가져오는 중 오류 발생: {e}")
        return None


# --- [개선 1] 상세하고 구조화된 프롬프트 ---
def create_youtube_script_prompt_v2(summary, topic, duration_minutes=10):
    """
    LLM이 길고 흥미로운 대본을 작성하도록 유도하는 구조화된 프롬프트
    """
    target_length = duration_minutes * 250
    prompt = f"""
    당신은 100만 구독자를 보유한 '유튜브 역사 스토리텔러'입니다.
    주어진 주제와 핵심 정보를 바탕으로, 청중이 시간 가는 줄 모르고 빠져들 만한 **{duration_minutes}분 분량(약 {target_length}자 이상)의 유튜브 영상 대본**을 작성해주세요.

    ### 주제: {topic}

    ### 핵심 정보:
    {summary}

    ### 대본 필수 구성 요소 및 작성 지침:
    1.  **강렬한 도입부 (약 10%):**
        - 주제와 관련된 충격적인 사실이나, 시청자의 허를 찌르는 질문으로 시작해주세요.
        - "상상해보셨나요?", "만약 ~라면 어땠을까요?" 와 같은 화법을 사용해 시청자의 호기심을 즉시 자극해야 합니다.

    2.  **흥미진진한 본문 (약 80%):**
        - 핵심 정보를 바탕으로 이야기를 시간 순서나 주제별로 자연스럽게 연결해주세요.
        - 최소 3개 이상의 소주제로 단락을 나누어 내용을 전개해주세요. (예: 출생의 비밀, 천재성의 발현, 세종과의 만남, 위대한 발명품들, 갑작스러운 몰락 등)
        - 역사적 사실에 당신의 상상력과 해석을 더해, 마치 한 편의 영화를 보는 듯 생생하게 묘사해주세요.
        - 전문용어는 최대한 피하고, 비유나 예시를 들어 중학생도 이해할 수 있도록 쉽게 설명해주세요.

    3.  **인상적인 마무리 (약 10%):**
        - 전체 이야기를 요약하며 주제의 의미를 다시 한번 강조해주세요.
        - 시청자에게 생각할 거리를 던져주거나, 감동적인 메시지로 여운을 남겨주세요.
        - "오늘 이야기가 흥미로우셨다면 구독과 좋아요 잊지 마세요!" 와 같은 채널 성장을 위한 멘트도 자연스럽게 포함해주세요.

    ### 어조 및 스타일:
    - 딱딱한 설명조가 아닌, 친한 친구나 선배가 흥미로운 옛날이야기를 들려주는 듯한 **친근하고 열정적인 말투**를 사용해주세요.
    - 모든 문장은 입으로 소리 내어 읽기 편하도록 간결하게 작성해주세요.

    ### 출력 형식:
    - **오직 영상에서 말할 대본 텍스트만** 다른 설명 없이 바로 출력해주세요.
    """
    return prompt.strip()


def split_text_into_chunks(text, max_length=200):
    """안정적인 TTS 생성을 위해 긴 텍스트를 문장 단위로 분할"""
    # 문장 분리 (마침표, 물음표, 느낌표 기준)
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# --- 메인 로직 ---
def main():
    # --- 설정 변수 ---
    KEYWORD = "장영실"
    OLLAMA_MODEL = "gemma3:4b"  # 긴 글 생성에 더 유리한 모델 추천
    OUTPUT_SCRIPT_FILE = f"{KEYWORD}_script.txt"
    OUTPUT_AUDIO_FILE = f"{KEYWORD}_audio_bark_final.wav"
    BARK_SAMPLE_RATE = 24_000
    SILENCE_DURATION_MS = 300  # 문장 사이 쉼 길이 (밀리초)

    # --- [개선 2] GPU(CUDA) 사용 가능 여부 확인 ---
    if torch.cuda.is_available():
        print("✅ CUDA(GPU) 사용이 가능합니다. Bark 모델을 GPU에 로딩합니다.")
        # Bark는 자동으로 GPU를 사용하지만, 명시적으로 장치를 설정할 수도 있습니다.
        # from bark.generation import torch
        # torch.cuda.set_device(0)
    else:
        print(
            "⚠️ 경고: CUDA(GPU)를 사용할 수 없습니다. CPU로 음성 합성을 진행하며, 매우 오랜 시간이 소요될 수 있습니다."
        )

    # --- 1. Bark 모델 로딩 ---
    print("🔊 Bark TTS 모델을 로딩합니다... (첫 실행 시 시간이 소요될 수 있습니다)")
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

    print(
        f"🤖 Ollama({OLLAMA_MODEL})에 대본 생성을 요청합니다... (시간이 걸릴 수 있습니다)"
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": create_youtube_script_prompt_v2(wiki_summary, KEYWORD),
                }
            ],
        )
        generated_script = response["message"]["content"]
        cleaned_script = clean_text_for_tts(generated_script)

        with open(OUTPUT_SCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(cleaned_script)
        print(f"💾 대본을 '{OUTPUT_SCRIPT_FILE}' 파일로 저장했습니다.")

        # --- [개선 3] 텍스트를 문장 단위로 분할 (Chunking) ---
        script_chunks = split_text_into_chunks(cleaned_script)
        if not script_chunks:
            print("🚨 대본에서 문장을 찾을 수 없습니다. 처리할 내용이 없습니다.")
            return

        print(
            f"🎤 Bark 음성 합성을 시작합니다. 총 {len(script_chunks)}개의 문장을 처리합니다."
        )

        # --- 3. 음성 합성 (Bark) 및 파일 병합 ---
        full_audio_array = np.array([], dtype=np.int16)
        # 문장 사이에 넣을 조용한 구간 생성
        silence = np.zeros(
            int(SILENCE_DURATION_MS / 1000 * BARK_SAMPLE_RATE), dtype=np.int16
        )

        start_time = time.time()
        for i, chunk in enumerate(script_chunks):
            print(f'   - 문장 ({i+1}/{len(script_chunks)}) 생성 중: "{chunk[:40]}..."')
            # Bark는 한국어 생성을 위해 별도 태그가 필요 없습니다.
            audio_array = generate_audio(chunk)
            full_audio_array = np.concatenate([full_audio_array, audio_array, silence])

        end_time = time.time()
        print(f"✅ 음성 합성 완료! (소요 시간: {end_time - start_time:.2f}초)")

        # --- 4. 오디오 파일 저장 ---
        print(f"💾 생성된 오디오를 '{OUTPUT_AUDIO_FILE}' 파일로 저장합니다...")
        write_wav(OUTPUT_AUDIO_FILE, BARK_SAMPLE_RATE, full_audio_array)

        print(f"🎉 최종 성공! '{OUTPUT_AUDIO_FILE}' 음성 파일이 생성되었습니다.")

    except Exception as e:
        print(f"🚨 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
# %%
