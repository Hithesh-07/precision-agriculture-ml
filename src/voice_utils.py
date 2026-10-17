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
<div id="voice-container" style="font-family:Segoe UI, sans-serif;padding:6px;max-width:100%;">

  <div id="status-bar" style="
    background:#F2F8F4;border:1px solid #1F7A4D;border-radius:8px;
    padding:10px 14px;margin-bottom:12px;font-size:13px;color:#14532D;font-weight:500;">
    🎙️ Voice feature ready. Click <strong>Speak</strong> to begin.
  </div>

  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
    <button id="btn-mic" onclick="startRecognition()" style="
      background:#1F7A4D;color:white;border:none;border-radius:8px;
      padding:9px 20px;font-size:14px;font-weight:600;cursor:pointer;display:inline-flex;align-items:center;gap:6px;">
      🎤 Speak
    </button>
    <button id="btn-stop" onclick="stopRecognition()" style="
      background:#DC2626;color:white;border:none;border-radius:8px;
      padding:9px 20px;font-size:14px;font-weight:600;cursor:pointer;display:none;align-items:center;gap:6px;">
      ⏹ Stop Recording
    </button>
    <button id="btn-copy" onclick="copyTranscript()" style="
      background:#FFFFFF;color:#1F7A4D;border:1px solid #1F7A4D;border-radius:8px;
      padding:9px 16px;font-size:13px;font-weight:600;cursor:pointer;display:none;">
      📋 Copy Text
    </button>
    <span id="mic-indicator" style="display:none;font-size:12px;color:#EA580C;font-weight:600;">
      🔴 Listening...
    </span>
  </div>

  <div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:4px;">Recognised Speech Transcript:</div>
  <div id="transcript-box" style="
    background:#FFFFFF;border:1.5px solid #D1E7DA;border-radius:8px;
    padding:12px 14px;min-height:55px;font-size:14px;color:#1F2937;margin-bottom:8px;
    white-space:pre-wrap;line-height:1.4;">(recognised text will appear here as you speak)</div>

  <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#6B7280;flex-wrap:wrap;gap:6px;">
    <div>Recognition Language: <strong id="lang-display" style="color:#1F7A4D;">{LANG_CODE}</strong></div>
    <div id="browser-check" style="font-size:11px;"></div>
  </div>

  <div style="margin-top:10px;padding:8px 12px;background:#F9FAFB;border-radius:6px;font-size:11px;color:#4B5563;">
    💡 <strong>Tip:</strong> Say agricultural terms like <em>"Wheat in Punjab Kharif season 5000 hectares 1200 rainfall"</em>.
  </div>
</div>

<script>
var recognition = null;
var currentLang = '{LANG_CODE}';
var fullTranscript = '';

// Check browser support immediately
if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
  setStatus('⚠️ Speech recognition API is not supported in this browser. Please use Google Chrome or Microsoft Edge.', '#DC2626');
  document.getElementById('browser-check').innerHTML = '❌ Browser unsupported (use Chrome/Edge)';
  document.getElementById('btn-mic').disabled = true;
  document.getElementById('btn-mic').style.opacity = '0.5';
} else {
  document.getElementById('browser-check').innerHTML = '✅ Browser supported';
}

function setStatus(msg, colour) {
  var bar = document.getElementById('status-bar');
  bar.innerHTML = msg;
  bar.style.borderColor = colour || '#1F7A4D';
  bar.style.color = colour === '#DC2626' ? '#7F1D1D' : colour === '#EA580C' ? '#9A3412' : '#14532D';
  bar.style.background = colour === '#DC2626' ? '#FEF2F2' : colour === '#EA580C' ? '#FFFBEB' : '#F2F8F4';
}

function startRecognition() {
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    setStatus('❌ Speech recognition not available.', '#DC2626');
    return;
  }

  try {
    recognition = new SpeechRecognition();
    recognition.lang = currentLang;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = function() {
      setStatus('🎙️ <strong>Microphone active.</strong> Speak now into your microphone...', '#EA580C');
      document.getElementById('btn-mic').style.display = 'none';
      document.getElementById('btn-stop').style.display = 'inline-flex';
      document.getElementById('mic-indicator').style.display = 'inline-block';
      document.getElementById('transcript-box').style.borderColor = '#1F7A4D';
    };

    recognition.onresult = function(event) {
      var interim = '';
      var final = '';
      for (var i = 0; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          final += event.results[i][0].transcript + ' ';
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      var display = (final + interim).trim();
      if (display) {
        fullTranscript = display;
        document.getElementById('transcript-box').innerText = display;
        document.getElementById('btn-copy').style.display = 'inline-flex';
      }
    };

    recognition.onerror = function(event) {
      var err = event.error;
      var msg = 'Error: ' + err;
      if (err === 'not-allowed' || err === 'permission-denied') {
        msg = '🔒 Microphone permission was denied. Click the lock/camera icon in your browser address bar to allow microphone access.';
      } else if (err === 'no-speech') {
        msg = '🔇 No speech detected. Please speak louder or check microphone volume.';
      } else if (err === 'audio-capture') {
        msg = '🎙️ No microphone was detected on your system. Please connect an audio input device.';
      } else if (err === 'network') {
        msg = '🌐 Network error connecting to speech recognition service.';
      }
      setStatus(msg, '#DC2626');
      resetButtons();
    };

    recognition.onend = function() {
      if (fullTranscript) {
        setStatus('✅ <strong>Speech captured!</strong> Copy the text above to apply to form.', '#1F7A4D');
      } else {
        setStatus('🎙️ Recording finished. Click Speak to try again.', '#1F7A4D');
      }
      resetButtons();
    };

    recognition.start();
  } catch (e) {
    setStatus('❌ Could not start speech recognition: ' + e.message, '#DC2626');
    resetButtons();
  }
}

function stopRecognition() {
  if (recognition) {
    try { recognition.stop(); } catch(e){}
  }
  resetButtons();
}

function resetButtons() {
  document.getElementById('btn-mic').style.display = 'inline-flex';
  document.getElementById('btn-stop').style.display = 'none';
  document.getElementById('mic-indicator').style.display = 'none';
}

function copyTranscript() {
  var text = fullTranscript || document.getElementById('transcript-box').innerText;
  if (!text || text.indexOf('(recognised') !== -1) return;
  navigator.clipboard.writeText(text).then(function() {
    setStatus('📋 <strong>Copied to clipboard!</strong> Paste it into the Recognized Text input or notes below.', '#1F7A4D');
  }).catch(function() {
    // fallback
    var ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    setStatus('📋 <strong>Copied to clipboard!</strong>', '#1F7A4D');
  });
}
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
