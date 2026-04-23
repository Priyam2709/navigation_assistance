(() => {
  const session = window.PASSENGER_SESSION;
  const speakButton = document.getElementById("speak-guidance");
  const stopButton = document.getElementById("stop-guidance");
  const routeMap = document.getElementById("route-map");
  const voiceStatus = document.getElementById("voice-status");
  let voices = [];

  const updateVoiceStatus = (message) => {
    if (voiceStatus) {
      voiceStatus.textContent = message;
    }
  };

  const loadVoices = () => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    voices = window.speechSynthesis.getVoices() || [];
  };

  const pickVoice = () => {
    if (!voices.length) {
      return null;
    }
    return voices.find((voice) => /^en(-|_)/i.test(voice.lang)) || voices[0];
  };

  const speakInstructions = () => {
    if (!session?.route?.instructions?.length) {
      updateVoiceStatus("No route instructions are available for this session.");
      return;
    }
    if (!("speechSynthesis" in window)) {
      updateVoiceStatus("Speech synthesis is not available in this browser. Use the on-screen instructions.");
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(
      `Station ${session.station_name}. Current zone: ${session.current_zone_label}. Destination: ${session.route.destination_label}. ` +
        session.route.instructions.join(" ")
    );
    const voice = pickVoice();
    if (voice) {
      utterance.voice = voice;
    }
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.onstart = () => updateVoiceStatus("Guidance is being spoken aloud.");
    utterance.onend = () => updateVoiceStatus("Guidance finished. Press Speak Guidance to replay.");
    utterance.onerror = () => updateVoiceStatus("Speech playback could not be completed. Read the on-screen instructions.");
    window.speechSynthesis.speak(utterance);
  };

  const stopInstructions = () => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    window.speechSynthesis.cancel();
    updateVoiceStatus("Voice guidance stopped.");
  };

  const renderRoute = () => {
    if (!routeMap) {
      return;
    }

    const polyline = routeMap.querySelector("#route-polyline");
    const routePoints = JSON.parse(routeMap.dataset.route || "[]");
    if (polyline && routePoints.length) {
      polyline.setAttribute(
        "points",
        routePoints.map((point) => `${point.x},${point.y}`).join(" ")
      );
    }

    routeMap.querySelectorAll(".map-node").forEach((node) => {
      node.classList.remove("active", "destination");
      if (node.dataset.nodeId === routeMap.dataset.start) {
        node.classList.add("active");
      }
      if (node.dataset.nodeId === session.route.route_nodes[session.route.route_nodes.length - 1]) {
        node.classList.add("destination");
      }
    });
  };

  if (speakButton) {
    speakButton.addEventListener("click", speakInstructions);
  }

  if (stopButton) {
    stopButton.addEventListener("click", stopInstructions);
  }

  if ("speechSynthesis" in window) {
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }

  renderRoute();
})();
