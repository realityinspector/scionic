def is_palindrome(text: str) -> bool:
    """
    Check if a string is a palindrome, ignoring case, whitespace, and special characters.
    
    Args:
        text: The string to check for palindrome property
        
    Returns:
        bool: True if the string is a palindrome, False otherwise
        
    Examples:
        >>> is_palindrome("A man, a plan, a canal: Panama")
        True
        >>> is_palindrome("hello")
        False
        >>> is_palindrome("")
        True
    """
    # Handle empty string edge case
    if not text:
        return True
    
    # Normalize: convert to lowercase and keep only alphanumeric characters
    cleaned: str = ''.join(char.lower() for char in text if char.isalnum())
    
    # Handle edge case where cleaned string is empty (only special chars/spaces)
    if not cleaned:
        return True
    
    # Compare with reverse
    return cleaned == cleaned[::-1]


# Test cases
if __name__ == "__main__":
    test_cases: list[tuple[str, bool]] = [
        # Basic cases
        ("racecar", True),
        ("hello", False),
        
        # Empty string
        ("", True),
        
        # Case sensitivity
        ("RaceCar", True),
        ("A", True),
        
        # Whitespace and special characters
        ("A man, a plan, a canal: Panama", True),
        ("Was it a car or a cat I saw?", True),
        ("12321", True),
        ("12345", False),
        
        # Only special characters
        ("!!!???###", True),
        ("   ", True),
        
        # Mixed cases
        ("A1b2B1a", True),
        ("Able was I ere I saw Elba", True),
        ("Madam, I'm Adam", True),
        
        # Non-palindromes
        ("python", False),
        ("hello world", False),
    ]
    
    print("Running palindrome checker tests...\n")
    passed: int = 0
    failed: int = 0
    
    for text, expected in test_cases:
        result: bool = is_palindrome(text)
        status: str = "PASS" if result == expected else "FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        # Display with safe repr for clarity
        display_text: str = repr(text) if len(text) <= 50 else repr(text[:47] + "...")
        print(f"{status}: is_palindrome({display_text}) -> {result}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
