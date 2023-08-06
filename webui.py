import gradio as gr
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

def inference(checkpoint_path, audio_path, output_midi_path, device):
    checkpoint_path = checkpoint_path.name
    device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
 
    # Load audio
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device=device, checkpoint_path=checkpoint_path)

    # Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe(audio, output_midi_path)
    return output_midi_path

webui = gr.Interface(
    fn=inference,
    inputs=[
        gr.File(file_types=['.pth']),
        gr.Audio(type='filepath'),
        gr.Textbox(value='cut_liszt.mid'),
        gr.Radio(['cpu', 'cuda'])],
    outputs=['file'],
    title='WebUI for Piano Transcription'
)

if __name__ == '__main__' :
    webui.launch()
