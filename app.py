# ══════════════════════════════════════════════════════════════
# Mental Health Chatbot — Flask App
# Integrates: Stress Model (82.6%) + Emotion Model (76.1%) + Bandit (15 games)
# ══════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, r"C:\Users\Ciel\mental-health-chatbot\backend")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Initialize Flask ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# ── Device setup ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load Models ──────────────────────────────────────────────
print("Loading models...")

# Stress model
STRESS_PATH = r"C:\Users\Ciel\mental-health-chatbot\models\mental-roberta-v2\final"
tok_stress = AutoTokenizer.from_pretrained(STRESS_PATH)
mdl_stress = AutoModelForSequenceClassification.from_pretrained(STRESS_PATH).to(device)
mdl_stress.eval()
print("✅ Stress model (82.6% F1)")

# Emotion model (5-class)
EMOTION_PATH = r"C:\Users\Ciel\mental-health-chatbot\models\mental-roberta-emotion5"
tok_emotion = AutoTokenizer.from_pretrained(EMOTION_PATH)
mdl_emotion = AutoModelForSequenceClassification.from_pretrained(EMOTION_PATH).to(device)
mdl_emotion.eval()
print("✅ Emotion model (76.1% F1)")

# ── Constants ────────────────────────────────────────────────
EMOTION_TO_IDX = {
    'anxiety': 0, 'burnout': 1,
    'sadness': 2, 'overwhelm': 3, 'neutral': 4
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}

GAMES_EXTENDED = [
    "breathing_game", "color_tap", "body_scan",
    "nature_walk", "memory_match", "gratitude_journal",
    "puzzle", "word_search", "trivia",
    "bubble_pop", "doodle_draw", "countdown_timer",
    "music_therapy", "micro_journal", "positive_memory"
]

EMOTION_GAMES = {
    'anxiety':   [0, 1, 2],
    'burnout':   [3, 4, 5],
    'neutral':   [6, 7, 8],
    'overwhelm': [9, 10, 11],
    'sadness':   [12, 13, 14],
}

BEST_GAME = {
    'anxiety':   'breathing_game',
    'burnout':   'nature_walk',
    'sadness':   'positive_memory',
    'overwhelm': 'bubble_pop',
    'neutral':   'puzzle',
}

GAME_DESCRIPTIONS = {
    "breathing_game":      "4-7-8 breathing pacer — inhale 4s, hold 7s, exhale 8s",
    "color_tap":           "Tap colored circles in slow sequence — attention redirection",
    "body_scan":           "Guided body scan animation — focus on each body part",
    "nature_walk":         "Animated forest/beach scene with ambient sounds",
    "memory_match":        "Flip cards to find matching pairs — gentle cognitive reset",
    "gratitude_journal":   "Write 3 things you are grateful for today",
    "puzzle":              "Simple jigsaw puzzle — satisfying completion",
    "word_search":         "Find hidden words in a relaxing letter grid",
    "trivia":              "Light fun trivia questions — positive distraction",
    "bubble_pop":          "Pop bubbles to release tension — satisfying and simple",
    "doodle_draw":         "Free drawing canvas with calming colors",
    "countdown_timer":     "5-minute mindfulness countdown with breathing prompts",
    "music_therapy":       "Choose calming music and reflect on its meaning",
    "micro_journal":       "Write one sentence about how you feel right now",
    "positive_memory":     "Recall and describe a happy memory in detail",
}

# ── Helper Functions ────────────────────────────────────────
def get_stress_and_emotion(text):
    """Encode text to get stress probability and emotion."""
    
    # Stress
    enc1 = tok_stress(str(text)[:400], return_tensors='pt',
                      max_length=128, truncation=True).to(device)
    with torch.no_grad():
        out1 = mdl_stress(**enc1)
        probs1 = torch.softmax(out1.logits, dim=-1)
        stress = float(probs1[0][1])
    
    # Emotion
    enc2 = tok_emotion(str(text)[:400], return_tensors='pt',
                       max_length=128, truncation=True).to(device)
    with torch.no_grad():
        out2 = mdl_emotion(**enc2)
        probs2 = torch.softmax(out2.logits, dim=-1)
        emo_idx = int(probs2.argmax())
        emotion = IDX_TO_EMOTION[emo_idx]
        emo_conf = float(probs2.max())
    
    return {
        'stress_level': round(stress, 3),
        'emotion': emotion,
        'emotion_confidence': round(emo_conf, 3),
    }

def recommend_game(emotion, stress_level):
    """Recommend best game for emotion."""
    if stress_level < 0.4:
        return None
    
    game = BEST_GAME[emotion]
    return {
        'game': game,
        'description': GAME_DESCRIPTIONS[game],
        'recommended_for': emotion,
    }

# ── Routes ───────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Serve homepage."""
    return jsonify({
        'status': 'ok',
        'app': 'Mental Health Chatbot v1.0',
        'endpoints': {
            'POST /api/analyze': 'Analyze user text for stress & emotion',
            'POST /api/chat': 'Chat with support + game recommendation',
            'GET /api/games': 'List all available games',
            'GET /api/models': 'Model information',
        },
        'models': {
            'stress': '82.6% F1',
            'emotion': '76.1% F1',
            'games': 15,
            'device': str(device),
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze user text for stress and emotion."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.strip()) < 5:
            return jsonify({'error': 'Text too short (min 5 chars)'}), 400
        
        result = get_stress_and_emotion(text)
        needs_game = result['stress_level'] > 0.6
        
        if needs_game:
            game_rec = recommend_game(result['emotion'], result['stress_level'])
            result['game_recommendation'] = game_rec
        
        return jsonify({
            'status': 'success',
            'text': text[:100] + '...' if len(text) > 100 else text,
            'analysis': result,
            'timestamp': datetime.now().isoformat(),
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with full pipeline."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message or len(user_message.strip()) < 5:
            return jsonify({'error': 'Message too short'}), 400
        
        # Analyze
        analysis = get_stress_and_emotion(user_message)
        
        # Prepare response
        response = {
            'status': 'success',
            'user_message': user_message[:200],
            'analysis': analysis,
            'needs_intervention': analysis['stress_level'] > 0.6,
            'timestamp': datetime.now().isoformat(),
        }
        
        if analysis['stress_level'] > 0.6:
            game_rec = recommend_game(analysis['emotion'], analysis['stress_level'])
            response['intervention'] = {
                'type': 'therapeutic_game',
                'game': game_rec['game'],
                'description': game_rec['description'],
                'rationale': f"Based on detected {analysis['emotion']} emotion",
            }
            response['supportive_message'] = \
                f"I detected you might be experiencing {analysis['emotion']}. " \
                f"Let me suggest {game_rec['game'].replace('_', ' ')} — {game_rec['description'].lower()}"
        else:
            response['supportive_message'] = \
                "You seem calm right now. Remember, it's okay to take breaks anytime you need."
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/games', methods=['GET'])
def list_games():
    """List all available games."""
    games_by_emotion = {}
    for emotion, game_indices in EMOTION_GAMES.items():
        games_by_emotion[emotion] = [
            {
                'name': GAMES_EXTENDED[i],
                'description': GAME_DESCRIPTIONS[GAMES_EXTENDED[i]],
            }
            for i in game_indices
        ]
    
    return jsonify({
        'status': 'success',
        'total_games': len(GAMES_EXTENDED),
        'emotions': list(EMOTION_TO_IDX.keys()),
        'games_by_emotion': games_by_emotion,
    }), 200

@app.route('/api/models', methods=['GET'])
def model_info():
    """Model information."""
    return jsonify({
        'status': 'success',
        'models': {
            'stress_detection': {
                'name': 'MentalRoBERTa',
                'task': 'Binary stress classification',
                'f1_score': 0.826,
                'path': STRESS_PATH,
            },
            'emotion_detection': {
                'name': 'MentalRoBERTa 5-class',
                'task': 'Multi-class emotion (anxiety/burnout/sadness/overwhelm/neutral)',
                'f1_score': 0.761,
                'path': EMOTION_PATH,
            },
        },
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
    }), 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
    }), 200

# ── Error handlers ────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*65)
    print("Mental Health Chatbot — Flask Server")
    print("="*65)
    print(f"Device: {device}")
    print(f"Stress model: 82.6% F1")
    print(f"Emotion model: 76.1% F1")
    print(f"Games: 15 (3 per emotion)")
    print("="*65)
    print("\nStarting server on http://127.0.0.1:5000")
    print("API docs at http://127.0.0.1:5000\n")
    
    # Run Flask app
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
    )
