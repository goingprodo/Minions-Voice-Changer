import gradio as gr
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from scipy.io import wavfile
import tempfile
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class VoiceConverter:
    def __init__(self):
        self.sample_rate = 22050
    
    def pitch_shift(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """피치 변환 (음높이 조절)"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    
    def formant_shift(self, audio: np.ndarray, sr: int, shift_factor: float) -> np.ndarray:
        """포만트 변환 (음성 특성 조절)"""
        # STFT를 사용한 스펙트럼 조작
        stft = librosa.stft(audio, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 주파수 축 변환
        freq_bins = magnitude.shape[0]
        new_magnitude = np.zeros_like(magnitude)
        
        for i in range(freq_bins):
            new_freq = int(i * shift_factor)
            if new_freq < freq_bins:
                new_magnitude[new_freq] = magnitude[i]
        
        # 역변환
        new_stft = new_magnitude * np.exp(1j * phase)
        return librosa.istft(new_stft, hop_length=512)
    
    def apply_female_characteristics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """여성 음성 특성 적용"""
        # 고주파 강조 필터
        nyquist = sr // 2
        high_freq = 2000 / nyquist
        b, a = scipy.signal.butter(2, high_freq, btype='high')
        enhanced_audio = scipy.signal.filtfilt(b, a, audio)
        
        # 원본과 필터링된 오디오 믹싱
        return 0.7 * audio + 0.3 * enhanced_audio
    
    def convert_to_female_voice(
        self, 
        audio_path: str, 
        pitch_shift_semitones: float = 4.0,
        formant_shift_factor: float = 1.2,
        brightness: float = 1.1
    ) -> str:
        """남성 음성을 여성 음성으로 변환"""
        try:
            # 오디오 파일 로드
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 1. 피치 변환 (음높이를 높임)
            audio = self.pitch_shift(audio, sr, pitch_shift_semitones)
            
            # 2. 포만트 변환 (음성 특성 조절)
            audio = self.formant_shift(audio, sr, formant_shift_factor)
            
            # 3. 여성 음성 특성 적용 (밝기 조절)
            if brightness > 1.0:
                audio = self.apply_female_characteristics(audio, sr)
            
            # 4. 음량 정규화
            audio = librosa.util.normalize(audio)
            
            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio, sr)
            
            return temp_file.name
            
        except Exception as e:
            print(f"변환 중 오류 발생: {str(e)}")
            return None

# VoiceConverter 인스턴스 생성
converter = VoiceConverter()

def process_voice(
    audio_input,
    pitch_shift: float,
    formant_shift: float,
    brightness: float
) -> Tuple[Optional[str], str]:
    """음성 처리 메인 함수"""
    
    if audio_input is None:
        return None, "⚠️ 음성을 먼저 녹음하거나 업로드해주세요."
    
    try:
        # Gradio에서 받은 오디오 데이터 처리
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input
            
            # 임시 파일로 저장
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wavfile.write(temp_input.name, sample_rate, audio_data)
            input_path = temp_input.name
        else:
            input_path = audio_input
        
        # 음성 변환 실행
        output_path = converter.convert_to_female_voice(
            input_path, 
            pitch_shift_semitones=pitch_shift,
            formant_shift_factor=formant_shift,
            brightness=brightness
        )
        
        if output_path:
            return output_path, "✅ 변환이 완료되었습니다! 아래에서 결과를 들어보세요."
        else:
            return None, "❌ 변환 중 오류가 발생했습니다."
            
    except Exception as e:
        return None, f"❌ 처리 중 오류 발생: {str(e)}"

def reset_controls():
    """컨트롤 초기화"""
    return 4.0, 1.2, 1.1

# Gradio 인터페이스 생성
def create_interface():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(
        title="🎤 AI 음성 변환기",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main-header {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .control-panel {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
        """
    ) as interface:
        
        # 헤더
        gr.HTML("""
        <div class="main-header">
            <h1>🎤 AI 음성 변환기</h1>
            <p>남성 음성을 자연스러운 여성 음성으로 변환해보세요</p>
        </div>
        """)
        
        with gr.Row():
            # 입력 섹션
            with gr.Column(scale=1):
                gr.HTML("<div class='control-panel'>")
                gr.Markdown("### 📥 음성 입력")
                
                audio_input = gr.Audio(
                    label="음성 녹음 또는 파일 업로드",
                    type="numpy",
                    format="wav"
                )
                
                gr.Markdown("### ⚙️ 변환 설정")
                
                pitch_shift = gr.Slider(
                    minimum=0,
                    maximum=8,
                    step=0.5,
                    value=4.0,
                    label="피치 변환 (반음)",
                    info="높을수록 더 높은 음성"
                )
                
                formant_shift = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    step=0.1,
                    value=1.2,
                    label="포만트 변환",
                    info="음성 특성 조절"
                )
                
                brightness = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    step=0.1,
                    value=1.1,
                    label="밝기 조절",
                    info="여성 음성 특성 강화"
                )
                
                with gr.Row():
                    convert_btn = gr.Button(
                        "✨ 여성 음성으로 변환", 
                        variant="primary",
                        size="lg"
                    )
                    reset_btn = gr.Button(
                        "🔄 설정 초기화",
                        variant="secondary"
                    )
                
                gr.HTML("</div>")
            
            # 출력 섹션
            with gr.Column(scale=1):
                gr.HTML("<div class='control-panel'>")
                gr.Markdown("### 📤 변환 결과")
                
                status_output = gr.Textbox(
                    label="처리 상태",
                    value="음성을 입력하고 변환 버튼을 눌러주세요",
                    interactive=False
                )
                
                audio_output = gr.Audio(
                    label="변환된 음성",
                    type="filepath"
                )
                
                gr.Markdown("""
                ### 💡 사용 팁
                - **피치 변환**: 3-5 반음이 자연스러운 여성 음성 범위입니다
                - **포만트 변환**: 1.1-1.3 사이가 권장됩니다
                - **밝기 조절**: 1.0-1.2 사이로 설정하면 자연스럽습니다
                - 명확하고 또렷한 발음으로 녹음하면 더 좋은 결과를 얻을 수 있습니다
                """)
                
                gr.HTML("</div>")
        
        # 이벤트 핸들러
        convert_btn.click(
            fn=process_voice,
            inputs=[audio_input, pitch_shift, formant_shift, brightness],
            outputs=[audio_output, status_output]
        )
        
        reset_btn.click(
            fn=reset_controls,
            outputs=[pitch_shift, formant_shift, brightness]
        )
        
        # 예제 추가
        gr.Examples(
            examples=[
                [None, 4.0, 1.2, 1.1],  # 기본 설정
                [None, 3.5, 1.15, 1.0], # 자연스러운 설정
                [None, 5.0, 1.3, 1.2],  # 높은 음성
            ],
            inputs=[audio_input, pitch_shift, formant_shift, brightness],
        )
    
    return interface

# 메인 실행 함수
def main():
    """애플리케이션 실행"""
    print("🎤 AI 음성 변환기를 시작합니다...")
    print("필요한 라이브러리를 확인하는 중...")
    
    try:
        import librosa
        import soundfile
        print("✅ 모든 라이브러리가 준비되었습니다!")
    except ImportError as e:
        print(f"❌ 필요한 라이브러리가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치해주세요:")
        print("pip install gradio librosa soundfile scipy numpy")
        return
    
    # 인터페이스 생성 및 실행
    interface = create_interface()
    interface.launch(
        share=True,  # 외부 접근 가능한 링크 생성
        inbrowser=True,  # 자동으로 브라우저 열기
        server_name="0.0.0.0",  # 모든 네트워크에서 접근 가능
        server_port=7860,  # 포트 번호
        show_error=True  # 에러 메시지 표시
    )

if __name__ == "__main__":
    main()