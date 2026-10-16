"""
src/voice_utils.py
==================
Voice input/output helper — generates the browser-side JavaScript component
for speech recognition and speech synthesis.

Implementation notes:
- Uses browser Web Speech API (SpeechRecognition + SpeechSynthesis).
- Injected as an HTML component via Streamlit's st.components.v1.html().
- Supported browsers: Chrome, Edge (desktop). Limited Safari support.
  Firefox does not support SpeechRecognition as of 2025.
- Hindi (hi-IN) recognition and synthesis depend on available browser voices.
- Audio is processed by the browser's speech service (may be cloud-based).
- No audio data is stored in the Streamlit session or on disk.
"""

VOICE_COMPONENT_HTML = """
<div id="voice-container" style="font-family:sans-serif;padding:8px;">

  <div id="status-bar" style="
    background:#f0f8f0;border:1px solid #4CAF50;border-radius:6px;
    padding:8px 12px;margin-bottom:10px;font-size:13px;color:#2e7d32;">
    Voice feature ready. Click the microphone to speak.
  </div>

  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
    <button id="btn-mic" onclick="startRecognition()" style="
      background:#4CAF50;color:white;border:none;border-radius:6px;
      padding:10px 18px;font-size:15px;cursor:pointer;">
      🎤 Speak
    </button>
    <button id="btn-stop" onclick="stopRecognition()" style="
      background:#f44336;color:white;border:none;border-radius:6px;
      padding:10px 18px;font-size:15px;cursor:pointer;display:none;">
      ⏹ Stop
    </button>
  </div>

  <div id="transcript-box" style="
    background:white;border:1px solid #ccc;border-radius:6px;
    padding:10px;min-height:40px;font-size:14px;margin-bottom:10px;
    white-space:pre-wrap;">
    (recognised text will appear here)
  </div>

  <div id="lang-info" style="font-size:12px;color:#666;margin-bottom:8px;">
    Recognition language: <span id="lang-display">en-IN</span>
  </div>

  <div id="privacy-note" style="font-size:11px;color:#888;margin-top:6px;">
    ⚠️ Note: Browser speech recognition may send audio to an external server
    depending on your browser. No audio is stored by this application.
  </div>
</div>

<script>
var recognition = null;
var currentLang = '{LANG_CODE}';

function setStatus(msg, colour) {{
  var bar = document.getElementById('status-bar');
  bar.innerText = msg;
  bar.style.borderColor = colour || '#4CAF50';
  bar.style.background = colour === '#f44336' ? '#fff0f0' :
                          colour === '#ff9800' ? '#fff8e1' : '#f0f8f0';
}}

function startRecognition() {{
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
    setStatus('❌ Speech recognition is not supported in this browser. Try Chrome or Edge.', '#f44336');
    return;
  }}

  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = currentLang;
  recognition.continuous = false;
  recognition.interimResults = true;

  recognition.onstart = function() {{
    setStatus('🔴 Recording... Speak now.', '#ff9800');
    document.getElementById('btn-mic').style.display = 'none';
    document.getElementById('btn-stop').style.display = 'inline-block';
  }};

  recognition.onresult = function(event) {{
    var transcript = '';
    for (var i = event.resultIndex; i < event.results.length; i++) {{
      transcript += event.results[i][0].transcript;
    }}
    document.getElementById('transcript-box').innerText = transcript;
    if (event.results[event.results.length - 1].isFinal) {{
      setStatus('✅ Recognition complete. Review text above.', '#4CAF50');
      // Send transcript to Streamlit via postMessage
      window.parent.postMessage({{type: 'voice_transcript', text: transcript}}, '*');
    }}
  }};

  recognition.onerror = function(event) {{
    var msgs = {{
      'not-allowed': 'Microphone permission denied. Please allow microphone access.',
      'no-speech': 'No speech detected. Try speaking louder.',
      'audio-capture': 'No microphone found.',
      'network': 'Network error during speech recognition.',
    }};
    setStatus('❌ ' + (msgs[event.error] || 'Error: ' + event.error), '#f44336');
    resetButtons();
  }};

  recognition.onend = function() {{
    resetButtons();
  }};

  recognition.start();
}}

function stopRecognition() {{
  if (recognition) recognition.stop();
  resetButtons();
}}

function resetButtons() {{
  document.getElementById('btn-mic').style.display = 'inline-block';
  document.getElementById('btn-stop').style.display = 'none';
}}

// Update language display
document.getElementById('lang-display').innerText = currentLang;
</script>
"""

TTS_COMPONENT_HTML = """
<div id="tts-container" style="font-family:sans-serif;padding:8px;">
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
    <button onclick="speakText()" style="
      background:#1976D2;color:white;border:none;border-radius:6px;
      padding:10px 18px;font-size:15px;cursor:pointer;">
      🔊 Listen
    </button>
    <button onclick="stopSpeaking()" style="
      background:#666;color:white;border:none;border-radius:6px;
      padding:10px 18px;font-size:15px;cursor:pointer;">
      ⏹ Stop
    </button>
    <span id="tts-status" style="font-size:13px;color:#555;"></span>
  </div>
  <div style="font-size:11px;color:#888;margin-top:6px;">
    Uses browser text-to-speech. Hindi voice availability depends on your device.
  </div>
</div>

<script>
function speakText() {{
  if (!window.speechSynthesis) {{
    document.getElementById('tts-status').innerText =
      '❌ Speech synthesis not supported in this browser.';
    return;
  }}
  window.speechSynthesis.cancel();
  var text = `{TEXT}`;
  var lang = '{LANG_CODE}';
  var utter = new SpeechSynthesisUtterance(text);
  utter.lang = lang;

  // Try to find a matching voice
  var voices = window.speechSynthesis.getVoices();
  var match = voices.find(v => v.lang.startsWith(lang.split('-')[0]));
  if (match) {{
    utter.voice = match;
  }} else {{
    document.getElementById('tts-status').innerText =
      'ℹ️ No ' + lang + ' voice found; using default voice.';
  }}

  utter.onstart = function() {{
    document.getElementById('tts-status').innerText = '▶ Speaking...';
  }};
  utter.onend = function() {{
    document.getElementById('tts-status').innerText = '✅ Done.';
  }};
  utter.onerror = function(e) {{
    document.getElementById('tts-status').innerText = '❌ Error: ' + e.error;
  }};
  window.speechSynthesis.speak(utter);
}}

function stopSpeaking() {{
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  document.getElementById('tts-status').innerText = 'Stopped.';
}}
</script>
"""


def get_voice_input_html(lang_code: str = "en-IN") -> str:
    """Return the voice input HTML component with the given language code."""
    return VOICE_COMPONENT_HTML.replace("{LANG_CODE}", lang_code)


def get_tts_html(text: str, lang_code: str = "en-IN") -> str:
    """Return the TTS HTML component for the given text and language."""
    # Escape backticks and backslashes for JS template literal
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    return TTS_COMPONENT_HTML.replace("{TEXT}", safe_text).replace("{LANG_CODE}", lang_code)


def build_prediction_speech_text(
    crop: str,
    predicted_yield: float,
    model_name: str,
    lang: str = "en",
) -> str:
    """Build the spoken output text for a prediction result."""
    if lang == "hi":
        return (
            f"फसल {crop} के लिए अनुमानित उपज {predicted_yield:.2f} टन प्रति हेक्टेयर है। "
            f"यह अनुमान {model_name} मॉडल द्वारा प्रदान किया गया है। "
            f"यह एक अनुमान है, गारंटीकृत परिणाम नहीं।"
        )
    return (
        f"The predicted yield for {crop} is {predicted_yield:.2f} tonnes per hectare. "
        f"This estimate was produced by the {model_name} model. "
        f"This is an estimate based on historical data, not a guaranteed outcome."
    )
