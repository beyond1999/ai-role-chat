const API_BASE = "http://localhost:8000"; // FastAPI 后端
let mediaRecorder;
let chunks = [];
const btnRecord = document.getElementById("btnRecord");
const btnStop = document.getElementById("btnStop");
const btnSendText = document.getElementById("btnSendText");
const statusEl = document.getElementById("status");
const asrTextEl = document.getElementById("asrText");
const llmTextEl = document.getElementById("llmText");
const player = document.getElementById("player");

async function startRecord() {
  chunks = [];
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // webm/opus 浏览器支持最好；后端用 ffmpeg 能解
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = onStopRecord;
  mediaRecorder.start();
  btnRecord.disabled = true;
  btnStop.disabled = false;
  statusEl.textContent = "录音中… 点击停止";
}

function stopRecord() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    btnStop.disabled = true;
  }
}

async function onStopRecord() {
  btnRecord.disabled = false;
  statusEl.textContent = "上传识别中…";
  const blob = new Blob(chunks, { type: "audio/webm" });
  const fd = new FormData();
  // 给个文件名，后端会保存
  fd.append("file", blob, "record.webm");

  // 1) ASR
  const asrRes = await fetch(`${API_BASE}/asr`, { method: "POST", body: fd });
  const asrData = await asrRes.json();
  asrTextEl.textContent = asrData.text || "(未识别到文本)";
  statusEl.textContent = "LLM 生成中…";

  // 2) Chat
  const persona = document.getElementById("persona").value;
  const chatFd = new FormData();
  chatFd.append("user_text", asrData.text || "");
  chatFd.append("persona", persona);
  const chatRes = await fetch(`${API_BASE}/chat`, { method: "POST", body: chatFd });
  const chatData = await chatRes.json();
  llmTextEl.textContent = chatData.reply || "";
  statusEl.textContent = "合成语音中…";

  // 3) TTS
  const ttsFd = new FormData();
  ttsFd.append("text", chatData.reply || "");
  const ttsRes = await fetch(`${API_BASE}/tts`, { method: "POST", body: ttsFd });
  const ttsBlob = await ttsRes.blob();
  player.src = URL.createObjectURL(ttsBlob);
  player.play().catch(()=>{});
  statusEl.textContent = "完成 ✅";
}

async function sendText() {
  const text = document.getElementById("textInput").value.trim();
  if (!text) return;
  asrTextEl.textContent = text;
  statusEl.textContent = "LLM 生成中…";

  const persona = document.getElementById("persona").value;
  const chatFd = new FormData();
  chatFd.append("user_text", text);
  chatFd.append("persona", persona);
  const chatRes = await fetch(`${API_BASE}/chat`, { method: "POST", body: chatFd });
  const chatData = await chatRes.json();
  llmTextEl.textContent = chatData.reply || "";

  statusEl.textContent = "合成语音中…";
  const ttsFd = new FormData();
  ttsFd.append("text", chatData.reply || "");
  const ttsRes = await fetch(`${API_BASE}/tts`, { method: "POST", body: ttsFd });
  const ttsBlob = await ttsRes.blob();
  player.src = URL.createObjectURL(ttsBlob);
  player.play().catch(()=>{});
  statusEl.textContent = "完成 ✅";
}

btnRecord.addEventListener("click", startRecord);
btnStop.addEventListener("click", stopRecord);
btnSendText.addEventListener("click", sendText);
