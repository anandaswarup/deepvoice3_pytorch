"""Frontend text processor for english"""

_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? "

symbols = [_pad, _eos, _unk] + list(_characters)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def _symbols_to_sequence(symbols):
    return [
        _symbol_to_id[s] if s in symbols else _symbol_to_id["_UNK_"]
        for s in symbols
    ]


def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text
        Args:
            text: string to convert to a sequence
        Returns:
            List of integers corresponding to the symbols in the text
    """
    text_seq = _symbols_to_sequence(text)

    # Append EOS token
    text_seq.append(_symbol_to_id["_EOS"])

    return text_seq


def num_chars():
    """Return the number of characters in the vocabulary
    """
    return len(_symbol_to_id)
