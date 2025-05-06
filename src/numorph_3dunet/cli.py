#!/usr/bin/env python
"""
Command-line interface for NuMorph 3DUnet.
"""

from numorph_3dunet.nuclei.generate_chunks import main

def predict():
    """
    Entry point for the prediction command.
    This function is called when running `numorph_3dunet.predict`
    """
    # Call the main function from generate_chunks
    main()

if __name__ == "__main__":
    predict()
