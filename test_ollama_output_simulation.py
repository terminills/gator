"""
Test to simulate actual Ollama output with ANSI codes and verify they're stripped.
"""

import sys
sys.path.insert(0, '/home/runner/work/gator/gator/src')

from backend.services.ai_models import strip_ansi_codes


def test_realistic_ollama_output():
    """
    Test with realistic Ollama CLI output that includes:
    - Spinner animations during thinking
    - Progress indicators
    - Cursor movements
    - Line clearing codes
    - Final text output
    """
    
    # Simulate what Ollama actually outputs when generating text
    # This is captured line-by-line from stdout
    ollama_lines = [
        # Initial spinner while loading model
        "\x1b[?2026h\x1b[?25l\x1b[1G⠋ Loading model...\x1b[K",
        "\x1b[?25h\x1b[?2026l",
        
        # Spinner during generation (various frames)
        "\x1b[?2026h\x1b[?25l\x1b[1G⠙ \x1b[K",
        "\x1b[?25h\x1b[?2026l",
        "\x1b[?2026h\x1b[?25l\x1b[1G⠹ \x1b[K",
        "\x1b[?25h\x1b[?2026l",
        "\x1b[?2026h\x1b[?25l\x1b[1G⠸ \x1b[K",
        "\x1b[?25h\x1b[?2026l",
        "\x1b[?2026h\x1b[?25l\x1b[1G⠼ \x1b[K",
        "\x1b[?25h\x1b[?2026l",
        
        # Clear line and show actual output
        "\x1b[2K\x1b[1G",
        
        # Actual generated text (this is what we want to keep)
        "Listen up! What's the problem you need help with?",
        "Don't waste my time with small talk.",
    ]
    
    print("=" * 70)
    print("Simulating Ollama CLI Output Processing")
    print("=" * 70)
    print()
    
    # Process each line as the code does
    cleaned_lines = []
    for i, line in enumerate(ollama_lines):
        clean = strip_ansi_codes(line)
        print(f"Line {i:2d}:")
        print(f"  Raw:     {repr(line[:60])}")
        print(f"  Cleaned: {repr(clean[:60])}")
        if clean:
            # Apply the same filtering logic as the actual code
            stripped = clean.strip()
            if len(stripped) <= 2 or "Loading model" in stripped:
                print(f"  Skipped: spinner or loading message")
                continue
            # Only keep non-empty lines after stripping
            cleaned_lines.append(clean)
        print()
    
    # Join lines as the code does
    final_output = "\n".join(cleaned_lines).strip()
    
    print("=" * 70)
    print("Final Output (as returned by API):")
    print("=" * 70)
    print(final_output)
    print()
    
    # Verify the output
    assert "Listen up!" in final_output, "Missing expected text"
    assert "Don't waste my time" in final_output, "Missing expected text"
    assert "\x1b" not in final_output, "Found ESC character in output!"
    assert "[?2026h" not in final_output, "Found control sequence in output!"
    assert "[K" not in final_output, "Found line clear code in output!"
    
    # Check that we don't have excessive spinner characters
    # Some might remain from incomplete clears, but not many
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spinner_count = sum(final_output.count(c) for c in spinner_chars)
    
    print(f"Spinner characters in final output: {spinner_count}")
    print(f"Total output length: {len(final_output)} characters")
    
    # Most spinner chars should be cleared by line clear codes
    assert spinner_count <= 3, f"Too many spinner characters in output: {spinner_count}"
    
    print()
    print("=" * 70)
    print("✅ All checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_realistic_ollama_output()
