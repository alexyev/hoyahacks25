from treys import Card, Evaluator
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os

def convert_cv_to_treys(output_cards): 
    # converts ranks into treys interpretable symbols
    # might have to convert the names here, but when Ibrahim finishes
    rank_convert = {
        'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5',
        '6': '6', '7': '7', '8': '8', '9': '9', '10': 'T',
        'J': 'J', 'Q': 'Q', 'K': 'K'
    }

    # converts suits to treys interpretable symbols
    suit_convert = {
        'Hearts': 'h', 'Diamonds': 'd',
        'Clubs': 'c', 'Spades': 's' 
    }

    hand_cards = []
    table_cards = []
    
    for card in output_cards[hand_cards]: 
        # assume that we get outputs of card.rank, card.suit from the vision model
        rank = rank_convert[card.rank]
        suit = suit_convert[card.suit]
        new_card = f"{card.rank}{card.suit}"
        hand_cards.append(Card.new(new_card))

    for card in output_cards[table_cards]:
        rank = rank_convert[card.rank]
        suit = suit_convert[card.suit]
        new_card = f"{card.rank}{card.suit}"
        table_cards.append(Card.new(new_card))
    
    return hand_cards, table_cards

# pass in hand_cards, table_cards into this function
def evaluate_hand(board, hand):
    evaluator = Evaluator()
    hand_score = evaluator.evaluate(board, hand)
    hand_class = evaluator.get_rank_class(hand_rank)
    hand_desc = evaluator.class_to_string(hand_class)
    hand_perc = evaluator.get_five_card_rank_percentage(hand_class)
    return hand_perc, hand_desc

class TextToSpeech:
    def __init__(self):
        # Initialize the sound device
        self.device = sd.default.device
        # set default sample rate
        self.sample_rate = 22050 

    def speak(self, text):
        try: 
            # create a temp file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=false) as temp_file:
                temp_filename = temp_file.name

                # use e-speak for TTS
                subprocess.run(['espeak', '-w', temp_filename, text])

                # Read audio file 
                data, samplerate = sf.read(temp_filename)

                # play the audio
                sd.play(data, samplerate)
                sd.wait() # waits until the file is done playing

                # clean up temp filename
                os.unlink(temp_filename)

        except Exception as e: 
            print("Error in Text-to-Speech: {e}")


