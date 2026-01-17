// Melodi Hand - Dummy Backend Server
// Run with: node server.js

const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// ============================
// CHORD DATABASE
// ============================
const chordDatabase = {
  piano: [
    { chord: "C", chordType: "Major Triad", notes: ["C", "E", "G"] },
    { chord: "G", chordType: "Major Triad", notes: ["G", "B", "D"] },
    { chord: "Am", chordType: "Minor Triad", notes: ["A", "C", "E"] },
    { chord: "F", chordType: "Major Triad", notes: ["F", "A", "C"] },
    { chord: "Em", chordType: "Minor Triad", notes: ["E", "G", "B"] },
    { chord: "Dm", chordType: "Minor Triad", notes: ["D", "F", "A"] },
    { chord: "C7", chordType: "Dominant 7th", notes: ["C", "E", "G", "Bb"] },
    { chord: "Gmaj7", chordType: "Major 7th", notes: ["G", "B", "D", "F#"] }
  ],
  guitar: [
    { chord: "E", chordType: "Open Chord", notes: ["E", "B", "E", "G#", "B", "E"] },
    { chord: "A", chordType: "Open Chord", notes: ["A", "E", "A", "C#", "E"] },
    { chord: "D", chordType: "Open Chord", notes: ["D", "A", "D", "F#"] },
    { chord: "G", chordType: "Open Chord", notes: ["G", "B", "D", "G", "B", "G"] },
    { chord: "C", chordType: "Open Chord", notes: ["C", "E", "G", "C", "E"] },
    { chord: "Em", chordType: "Minor Chord", notes: ["E", "B", "E", "G", "B", "E"] },
    { chord: "Am", chordType: "Minor Chord", notes: ["A", "E", "A", "C", "E"] },
    { chord: "Bm", chordType: "Barre Chord", notes: ["B", "F#", "B", "D", "F#", "B"] }
  ],
  drums: [
    { chord: "Rock", chordType: "Basic Beat", notes: ["Kick", "Snare", "HiHat"] },
    { chord: "Jazz", chordType: "Swing Pattern", notes: ["Ride", "Snare", "Kick"] },
    { chord: "Latin", chordType: "Clave Rhythm", notes: ["Conga", "Bongo", "Cowbell"] },
    { chord: "Funk", chordType: "Syncopated", notes: ["Kick", "Snare", "HiHat", "Tom"] },
    { chord: "Pop", chordType: "Four-on-floor", notes: ["Kick", "Clap", "HiHat"] },
    { chord: "Blues", chordType: "Shuffle", notes: ["Kick", "Snare", "Ride"] },
    { chord: "Reggae", chordType: "One Drop", notes: ["Kick", "Snare", "HiHat"] },
    { chord: "Metal", chordType: "Double Bass", notes: ["Kick", "Kick", "Snare", "HiHat"] }
  ]
};

// ============================
// SIMULATED SENSOR STATE
// ============================
let deviceConnected = false;
let currentInstrument = "piano";
let sensorData = {
  flexSensor1: 0,
  flexSensor2: 0,
  flexSensor3: 0,
  flexSensor4: 0,
  flexSensor5: 0,
  accelerometerX: 0,
  accelerometerY: 0,
  accelerometerZ: 0,
  timestamp: Date.now()
};

// Simulate sensor data changes
setInterval(() => {
  if (deviceConnected) {
    sensorData = {
      flexSensor1: Math.random() * 100,
      flexSensor2: Math.random() * 100,
      flexSensor3: Math.random() * 100,
      flexSensor4: Math.random() * 100,
      flexSensor5: Math.random() * 100,
      accelerometerX: (Math.random() - 0.5) * 10,
      accelerometerY: (Math.random() - 0.5) * 10,
      accelerometerZ: (Math.random() - 0.5) * 10,
      timestamp: Date.now()
    };
  }
}, 500);

// ============================
// API ENDPOINTS
// ============================

// Check device connection status
app.get('/api/device/status', (req, res) => {
  res.json({
    connected: deviceConnected,
    instrument: currentInstrument,
    timestamp: Date.now()
  });
});

// Connect device
app.post('/api/device/connect', (req, res) => {
  deviceConnected = true;
  console.log('Device connected');
  res.json({
    success: true,
    message: 'Device connected successfully',
    timestamp: Date.now()
  });
});

// Disconnect device
app.post('/api/device/disconnect', (req, res) => {
  deviceConnected = false;
  console.log('Device disconnected');
  res.json({
    success: true,
    message: 'Device disconnected',
    timestamp: Date.now()
  });
});

// Set instrument
app.post('/api/instrument/set', (req, res) => {
  const { instrument } = req.body;
  
  if (!['piano', 'guitar', 'drums'].includes(instrument)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid instrument. Must be: piano, guitar, or drums'
    });
  }
  
  currentInstrument = instrument;
  console.log(`Instrument changed to: ${instrument}`);
  
  res.json({
    success: true,
    instrument: currentInstrument,
    timestamp: Date.now()
  });
});

// Get current sensor data
app.get('/api/sensor/data', (req, res) => {
  if (!deviceConnected) {
    return res.status(503).json({
      success: false,
      error: 'Device not connected'
    });
  }
  
  res.json({
    success: true,
    data: sensorData
  });
});

// Detect gesture and return chord
app.get('/api/detect-gesture', (req, res) => {
  if (!deviceConnected) {
    return res.status(503).json({
      success: false,
      error: 'Device not connected'
    });
  }
  
  // Simulate gesture detection based on sensor data
  const chords = chordDatabase[currentInstrument];
  
  // Simple algorithm: use flex sensor average to determine chord
  const avgFlex = (
    sensorData.flexSensor1 +
    sensorData.flexSensor2 +
    sensorData.flexSensor3 +
    sensorData.flexSensor4 +
    sensorData.flexSensor5
  ) / 5;
  
  const chordIndex = Math.floor((avgFlex / 100) * chords.length);
  const detectedChord = chords[Math.min(chordIndex, chords.length - 1)];
  
  console.log(`Detected chord: ${detectedChord.chord} (${detectedChord.chordType})`);
  
  res.json({
    success: true,
    chord: detectedChord.chord,
    chordType: detectedChord.chordType,
    notes: detectedChord.notes,
    confidence: Math.random() * 0.3 + 0.7, // 70-100% confidence
    sensorSnapshot: {
      avgFlexValue: avgFlex.toFixed(2),
      timestamp: sensorData.timestamp
    }
  });
});

// Get all available chords for current instrument
app.get('/api/chords/list', (req, res) => {
  const instrument = req.query.instrument || currentInstrument;
  
  if (!chordDatabase[instrument]) {
    return res.status(400).json({
      success: false,
      error: 'Invalid instrument'
    });
  }
  
  res.json({
    success: true,
    instrument: instrument,
    chords: chordDatabase[instrument]
  });
});

// Calibrate device
app.post('/api/device/calibrate', (req, res) => {
  if (!deviceConnected) {
    return res.status(503).json({
      success: false,
      error: 'Device not connected'
    });
  }
  
  console.log('Calibrating device...');
  
  // Simulate calibration
  setTimeout(() => {
    console.log('Calibration complete');
  }, 2000);
  
  res.json({
    success: true,
    message: 'Calibration started',
    estimatedTime: 2000
  });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'online',
    version: '1.0.0',
    timestamp: Date.now()
  });
});

// ============================
// START SERVER
// ============================
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════╗
║     MELODI HAND - BACKEND SERVER      ║
╚════════════════════════════════════════╝

🚀 Server running on http://localhost:${PORT}

📡 Available Endpoints:
   GET  /api/health                - Health check
   GET  /api/device/status         - Device connection status
   POST /api/device/connect        - Connect device
   POST /api/device/disconnect     - Disconnect device
   POST /api/instrument/set        - Set instrument (body: {instrument})
   GET  /api/sensor/data           - Get current sensor readings
   GET  /api/detect-gesture        - Detect chord from gesture
   GET  /api/chords/list           - List all chords
   POST /api/device/calibrate      - Calibrate sensors

🎹 Current Instrument: ${currentInstrument}
🔌 Device Connected: ${deviceConnected}

Ready to receive requests...
  `);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n👋 Shutting down server...');
  process.exit(0);
});