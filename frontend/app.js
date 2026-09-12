const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const sampleBtn = document.getElementById('sampleBtn');
const messageInput = document.getElementById('message');
const resultCard = document.getElementById('resultCard');
const statusBadge = document.getElementById('statusBadge');
const intentBadge = document.getElementById('intentBadge');
const decisionBadge = document.getElementById('decisionBadge');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const replyText = document.getElementById('replyText');
const evidenceText = document.getElementById('evidenceText');
const reasonText = document.getElementById('reasonText');
const playbookText = document.getElementById('playbookText');

const sampleMessage = 'My package says delivered but I never received it';

function setStatus(text) {
  statusBadge.textContent = text;
}

function updateDecision(decision) {
  decisionBadge.textContent = decision;
  if (decision === 'escalate') {
    decisionBadge.classList.add('escalate');
  } else {
    decisionBadge.classList.remove('escalate');
  }
}

async function analyzeMessage() {
  const text = messageInput.value.trim();
  if (!text) {
    setStatus('Message needed');
    return;
  }

  setStatus('Analyzing...');
  sendBtn.disabled = true;

  try {
    const response = await fetch('/api/agent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: text })
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      setStatus('Error');
      replyText.textContent = data.error || 'Unable to analyze the customer message.';
      resultCard.classList.remove('hidden');
      return;
    }

    const out = data.result;
    const confidence = Math.round(Number(out.confidence || 0) * 100);

    intentBadge.textContent = out.intent || 'unknown';
    updateDecision(out.decision || 'auto-handle');
    confidenceBar.style.width = Math.min(Math.max(confidence, 2), 100) + '%';
    confidenceText.textContent = confidence + '%';
    replyText.textContent = out.reply || 'No reply generated.';
    reasonText.textContent = out.reason || 'No reason logged.';
    playbookText.textContent = out.playbook || 'No playbook available.';

    if (out.historical_evidence && out.historical_evidence.length > 0) {
      const top = out.historical_evidence[0];
      evidenceText.textContent = 'Intent: ' + top.intent + ' | Similarity: ' + top.similarity + ' | Customer text: ' + top.customer_text;
    } else {
      evidenceText.textContent = 'No historical evidence was found.';
    }

    resultCard.classList.remove('hidden');
    setStatus('Complete');
  } catch (error) {
    setStatus('Error');
    replyText.textContent = 'The backend could not be reached at /api/agent.';
    resultCard.classList.remove('hidden');
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', analyzeMessage);
clearBtn.addEventListener('click', () => {
  messageInput.value = '';
  resultCard.classList.add('hidden');
});
sampleBtn.addEventListener('click', () => {
  messageInput.value = sampleMessage;
  analyzeMessage();
});
