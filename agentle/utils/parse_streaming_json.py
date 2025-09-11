import json
import re
from typing import Any, TypeVar, cast
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


def parse_streaming_json(potential_json: str | None, model: type[T]) -> T:
    """
    Parse a potentially incomplete or malformed JSON string and create a Pydantic model instance.

    This function is designed to handle streaming LLM responses that may contain partial JSON,
    surrounded by other text, or have common formatting errors.

    Args:
        potential_json (str): The input string that may contain JSON data
        model (Type[BaseModel]): The Pydantic model class to instantiate

    Returns:
        BaseModel: An instance of the model with available data filled in
    """
    if potential_json is None:
        return model()

    # print(f"parsing: {potential_json}")

    def find_json_boundaries(text: str) -> tuple[int, int]:
        """Find the start and potential end of JSON in the text."""

        def try_find_valid_json_at_position(
            start_pos: int, start_char: str
        ) -> tuple[int, int]:
            """Try to find valid JSON boundaries starting at a specific position."""
            if start_char == "{":
                open_char, close_char = "{", "}"
            else:
                open_char, close_char = "[", "]"

            depth = 0
            in_string = False
            escape_next = False

            for i in range(start_pos, len(text)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\" and in_string:
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == open_char:
                        depth += 1
                    elif char == close_char:
                        depth -= 1
                        if depth == 0:
                            return start_pos, i + 1

            # If we reach here, no matching closing brace found
            return start_pos, len(text)

        # Look for JSON start markers - try each occurrence
        positions: list[tuple[int, str]] = []

        # Find all potential JSON starts
        for i, char in enumerate(text):
            if char in ["{", "["]:
                positions.append((i, char))

        if not positions:
            return -1, -1

        # Try each position to find the first one that gives us valid boundaries
        for start_pos, start_char in positions:
            start_idx, end_idx = try_find_valid_json_at_position(start_pos, start_char)
            if end_idx > start_idx:
                # Found valid boundaries, return them
                return start_idx, end_idx

        # If no valid boundaries found, return the first position we found
        return positions[0][0], len(text)

    def fix_common_json_issues(json_str: str) -> str:
        """Fix common JSON formatting issues found in streaming data."""
        # Remove markdown code block markers
        json_str = re.sub(r"^```json\s*", "", json_str, flags=re.IGNORECASE)
        json_str = re.sub(r"```\s*$", "", json_str)

        # Remove any leading/trailing whitespace
        json_str = json_str.strip()

        # Fix missing closing quotes on string values (at the end)
        # Look for patterns like: "key": "value without closing quote
        json_str = re.sub(r':\s*"([^"]*?)(?:\s*[,}]|$)', r': "\1"', json_str)

        # Fix missing closing quotes for keys
        # Look for patterns like: "key without quotes:
        json_str = re.sub(r'"([^"]*?)(?=\s*:)', r'"\1"', json_str)

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Ensure the JSON has proper closing braces if it appears incomplete
        open_braces = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")

        # Add missing closing braces
        json_str += "}" * open_braces
        json_str += "]" * open_brackets

        return json_str

    def extract_data_manually(json_str: str) -> dict[str, Any]:
        """
        Manually extract key-value pairs from malformed JSON.
        This is a fallback when json.loads fails.
        """
        data = {}

        # Extract string key-value pairs with quoted keys
        # Pattern: "key": "value" or 'key': 'value'
        string_pattern = r'["\']([^"\']+)["\']:\s*["\']([^"\']*)["\']?'
        string_matches = re.findall(string_pattern, json_str)

        for key, value in string_matches:
            data[key] = value

        # Extract string key-value pairs with unquoted keys
        # Pattern: key: "value" (no quotes around key)
        unquoted_key_string_pattern = (
            r'([a-zA-Z_][a-zA-Z0-9_]*):\s*["\']([^"\']*)["\']?'
        )
        unquoted_string_matches = re.findall(unquoted_key_string_pattern, json_str)

        for key, value in unquoted_string_matches:
            if key not in data:  # Don't override if already found with quotes
                data[key] = value

        # Extract numeric key-value pairs with quoted keys
        # Pattern: "key": 123 or "key": 123.45
        numeric_pattern = r'["\']([^"\']+)["\']:\s*(-?\d+(?:\.\d+)?)'
        numeric_matches = re.findall(numeric_pattern, json_str)

        for key, value in numeric_matches:
            try:
                # Try to convert to int first, then float
                if "." in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value

        # Extract numeric key-value pairs with unquoted keys
        # Pattern: key: 123 (no quotes around key)
        unquoted_key_numeric_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*):\s*(-?\d+(?:\.\d+)?)"
        unquoted_numeric_matches = re.findall(unquoted_key_numeric_pattern, json_str)

        for key, value in unquoted_numeric_matches:
            if key not in data:  # Don't override if already found
                try:
                    if "." in value:
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value

        # Extract boolean key-value pairs with quoted keys
        # Pattern: "key": true/false
        bool_pattern = r'["\']([^"\']+)["\']:\s*(true|false)'
        bool_matches = re.findall(bool_pattern, json_str, re.IGNORECASE)

        for key, value in bool_matches:
            data[key] = value.lower() == "true"

        # Extract boolean key-value pairs with unquoted keys
        # Pattern: key: true/false (no quotes around key)
        unquoted_key_bool_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*):\s*(true|false)"
        unquoted_bool_matches = re.findall(
            unquoted_key_bool_pattern, json_str, re.IGNORECASE
        )

        for key, value in unquoted_bool_matches:
            if key not in data:  # Don't override if already found
                data[key] = value.lower() == "true"

        # Extract null key-value pairs with quoted keys
        # Pattern: "key": null
        null_pattern = r'["\']([^"\']+)["\']:\s*null'
        null_matches = re.findall(null_pattern, json_str, re.IGNORECASE)

        for key, _ in null_matches:
            data[key] = None

        # Extract null key-value pairs with unquoted keys
        # Pattern: key: null (no quotes around key)
        unquoted_key_null_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*):\s*null"
        unquoted_null_matches = re.findall(
            unquoted_key_null_pattern, json_str, re.IGNORECASE
        )

        for key, _ in unquoted_null_matches:
            if key not in data:  # Don't override if already found
                data[key] = None

        return data

    def safe_json_parse(json_str: str) -> dict[str, Any]:
        """
        Attempt to parse JSON with multiple fallback strategies.
        """
        if not json_str.strip():
            return {}

        # Strategy 1: Try parsing as-is
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
            else:
                return {}
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Try with common fixes
        try:
            fixed_json = fix_common_json_issues(json_str)
            result = json.loads(fixed_json)
            if isinstance(result, dict):
                return result
            else:
                return {}
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3: Try wrapping in braces if it looks like key-value pairs
        if ":" in json_str and not json_str.strip().startswith(("{", "[")):
            try:
                wrapped = "{" + json_str + "}"
                fixed_wrapped = fix_common_json_issues(wrapped)
                result = json.loads(fixed_wrapped)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 4: Manual extraction as last resort
        return extract_data_manually(json_str)

    # Main parsing logic
    try:
        # Step 1: Find JSON boundaries in the input
        start_idx, end_idx = find_json_boundaries(potential_json)

        if start_idx == -1:
            # No JSON found, return model with defaults
            return model()

        # Step 2: Extract the JSON portion
        json_portion = potential_json[start_idx:end_idx]

        # Step 3: Parse the JSON safely
        parsed_data = safe_json_parse(json_portion)

        # Step 4: Filter data to only include fields that exist in the model
        model_fields = set(cast(type[BaseModel], model).model_fields.keys())
        filtered_data = {k: v for k, v in parsed_data.items() if k in model_fields}

        # Step 5: Create and return the model instance
        return model(**filtered_data)

    except Exception as e:
        # Ultimate fallback - return model with defaults
        print(f"Warning: Failed to parse JSON ({e}), returning default model")
        return model()


# Test model for demonstration
if __name__ == "__main__":
    # Test models - moved inside main block as requested
    class User(BaseModel):
        """Test model with optional fields for demonstration."""

        name: str | None = Field(default=None, description="User's name")
        address: str | None = Field(default=None, description="User's address")
        age: int | None = Field(default=None, description="User's age")
        email: str | None = Field(default=None, description="User's email")
        is_active: bool | None = Field(
            default=None, description="Whether user is active"
        )

    class Profile(BaseModel):
        """More complex test model with nested data."""

        user_id: int | None = Field(default=None)
        username: str | None = Field(default=None)
        bio: str | None = Field(default=None)
        followers: int | None = Field(default=None)
        verified: bool | None = Field(default=None)

    class Address(BaseModel):
        """Address model for nested testing."""

        street: str | None = Field(default=None)
        city: str | None = Field(default=None)
        country: str | None = Field(default=None, description="Country code")
        postal_code: str | None = Field(default=None)

    class ContactInfo(BaseModel):
        """Contact information model."""

        email: str | None = Field(default=None)
        phone: str | None = Field(default=None)
        preferred_contact: str | None = Field(default=None)

    class CompanyInfo(BaseModel):
        """Company information model."""

        name: str | None = Field(default=None)
        department: str | None = Field(default=None)
        position: str | None = Field(default=None)
        years_experience: int | None = Field(default=None)

    class ComplexUser(BaseModel):
        """Complex nested user model for advanced testing."""

        id: int | None = Field(default=None)
        name: str | None = Field(default=None)
        age: int | None = Field(default=None)
        address: Address | None = Field(default=None)
        contact: ContactInfo | None = Field(default=None)
        company: CompanyInfo | None = Field(default=None)
        tags: list[str] | None = Field(default=None)
        metadata: dict[str, Any] | None = Field(default=None)
        is_premium: bool | None = Field(default=None)

    class TeamMember(BaseModel):
        """Team member for array testing."""

        name: str | None = Field(default=None)
        role: str | None = Field(default=None)
        active: bool | None = Field(default=None)

    class Project(BaseModel):
        """Project model with array of nested objects."""

        project_name: str | None = Field(default=None)
        description: str | None = Field(default=None)
        team_members: list[TeamMember] | None = Field(default=None)
        budget: float | None = Field(default=None)
        completed: bool | None = Field(default=None)

    print("Testing parse_streaming_json function\n" + "=" * 50)

    # Test cases covering various scenarios
    test_cases: list[dict[str, Any]] = [
        # Case 1: Partial JSON with incomplete string
        {
            "input": '```json\n{"name": "arthu',
            "description": "Partial JSON with incomplete string",
            "expected_name": "arthu",
        },
        # Case 2: Complete JSON with surrounding text
        {
            "input": 'Sure! Here\'s the JSON you requested: {"name": "Arthur", "address": "123 Main St"}',
            "description": "Complete JSON with surrounding text",
            "expected_name": "Arthur",
        },
        # Case 3: No JSON found
        {
            "input": "processing your request...",
            "description": "No JSON found in input",
            "expected_name": None,
        },
        # Case 4: JSON with missing closing brace
        {
            "input": '{"name": "John", "age": 30, "email": "john@example.com"',
            "description": "JSON missing closing brace",
            "expected_name": "John",
        },
        # Case 5: JSON with trailing comma
        {
            "input": '{"name": "Jane", "address": "456 Oak Ave", }',
            "description": "JSON with trailing comma",
            "expected_name": "Jane",
        },
        # Case 6: JSON with boolean and null values
        {
            "input": 'Here is the data: {"name": "Bob", "is_active": true, "address": null}',
            "description": "JSON with boolean and null values",
            "expected_name": "Bob",
        },
        # Case 7: Malformed JSON with missing quotes
        {
            "input": '{name: "Alice", age: 25}',
            "description": "JSON with missing quotes on keys",
            "expected_name": "Alice",
        },
        # Case 8: JSON in markdown code block
        {
            "input": """```json
            {
                "name": "Charlie",
                "address": "789 Pine St",
                "age": 35
            }
            ```""",
            "description": "JSON in markdown code block",
            "expected_name": "Charlie",
        },
        # Case 9: Severely broken JSON
        {
            "input": '{"name": "Dave", "add',
            "description": "Severely truncated JSON",
            "expected_name": "Dave",
        },
        # Case 10: Empty braces
        {
            "input": "The result is: {}",
            "description": "Empty JSON object",
            "expected_name": None,
        },
        # Case 11: Text before and after JSON
        {
            "input": 'Here is your data: {"name": "John", "age": 30} and that\'s all I have for now.',
            "description": "Text before and after complete JSON",
            "expected_name": "John",
        },
        # Case 12: Multiple JSON-like structures (should take first)
        {
            "input": 'First: {"name": "Alice"} Second: {"name": "Bob"} End.',
            "description": "Multiple JSON objects (should parse first)",
            "expected_name": "Alice",
        },
        # Case 13: Text after partial JSON
        {
            "input": 'Data: {"name": "Charlie", "age": 25 and then some more text here',
            "description": "Text after partial JSON (missing closing brace)",
            "expected_name": "Charlie",
        },
        # Case 14: JSON with nested braces in trailing text
        {
            "input": 'Result: {"name": "David"} Note: use {braces} carefully in text.',
            "description": "JSON followed by text containing braces",
            "expected_name": "David",
        },
        # Case 15: Long text before and after
        {
            "input": """
            I'm processing your request and here are the preliminary results.
            Please note that this data is subject to change based on further analysis.
            The current user information is as follows:
            {"name": "Emma Watson", "email": "emma@example.com", "is_active": true}
            
            Additional notes:
            - This data was last updated yesterday
            - Please verify the email address
            - Contact support if you need modifications
            """,
            "description": "Long explanatory text before and after JSON",
            "expected_name": "Emma Watson",
        },
        # Case 16: JSON array instead of object
        {
            "input": 'Here are the names: ["Alice", "Bob", "Charlie"] from our database.',
            "description": "JSON array with surrounding text (should return default)",
            "expected_name": None,  # Arrays don't match our object model
        },
        # Case 17: Nested JSON structure with text after
        {
            "input": '{"name": "Frank", "details": {"age": 30, "city": "NYC"}} End of data transmission.',
            "description": "Nested JSON with trailing text",
            "expected_name": "Frank",
        },
    ]

    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"Input: {repr(test_case['input'])}")

        try:
            result = parse_streaming_json(test_case["input"], User)
            print(f"Result: {result}")
            print(f"Name extracted: {result.name}")

            # Validate expected result if provided
            if "expected_name" in test_case:
                expected = test_case["expected_name"]
                actual = result.name
                status = "✅ PASS" if actual == expected else "❌ FAIL"
                print(f"Expected name: {expected}, Got: {actual} - {status}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

        print("-" * 40)

    print("\n" + "=" * 60)
    print("TESTING TEXT BEFORE/AFTER JSON")
    print("=" * 60)

    # Test cases specifically for text before and after JSON
    text_around_json_cases: list[dict[str, Any]] = [
        {
            "input": 'Processing... {"status": "complete", "name": "Test User"} Done!',
            "description": "Simple text before and after",
            "model": User,
            "expected": {"name": "Test User"},
        },
        {
            "input": """
            Hello! I found your user data. Here it is:
            
            {"name": "Sarah Connor", "age": 35, "email": "sarah@resistance.com"}
            
            Please let me know if you need any modifications to this information.
            Best regards!
            """,
            "description": "Multi-line text with JSON in the middle",
            "model": User,
            "expected": {"name": "Sarah Connor", "age": 35},
        },
        {
            "input": 'Before JSON: {"incomplete": "data", "name": "Partial" and after some text with } braces',
            "description": "Incomplete JSON with confusing braces in trailing text",
            "model": User,
            "expected": {"name": "Partial"},
        },
        {
            "input": 'Multiple objects: {"first": "data"} and {"name": "Second"} choose wisely',
            "description": "Multiple JSON objects (should pick first complete one)",
            "model": User,
            "expected": {},  # First object has no 'name' field
        },
        {
            "input": 'Error in first {invalid json} but then {"name": "Recovery"} works',
            "description": "Invalid JSON followed by valid JSON",
            "model": User,
            "expected": {},  # Should try first occurrence, which is invalid
        },
        {
            "input": """
            # Here's your user data in JSON format:
            
            ```json
            {
                "name": "Code Block User",
                "email": "code@example.com",
                "is_active": true
            }
            ```
            
            The above JSON contains all the relevant user information.
            """,
            "description": "JSON in markdown code block with surrounding text",
            "model": User,
            "expected": {"name": "Code Block User", "is_active": True},
        },
    ]

    for i, test_case in enumerate(text_around_json_cases, 1):
        print(f"Text Test {i}: {test_case['description']}")
        print(
            f"Input preview: {repr(test_case['input'][:80])}{'...' if len(test_case['input']) > 80 else ''}"
        )

        try:
            result = parse_streaming_json(test_case["input"], test_case["model"])
            print(f"Result: {result}")

            # Check expected values
            if "expected" in test_case:
                all_passed = True
                for field, expected_value in test_case["expected"].items():
                    actual_value = getattr(result, field, "MISSING_FIELD")
                    if actual_value == expected_value:
                        print(f"  ✅ {field}: {actual_value}")
                    else:
                        print(
                            f"  ❌ {field}: expected {expected_value}, got {actual_value}"
                        )
                        all_passed = False

                if not test_case[
                    "expected"
                ]:  # Empty expected dict means all should be None/default
                    all_default = all(
                        getattr(result, field_name, None) is None
                        for field_name in test_case["model"].model_fields.keys()
                    )
                    if all_default:
                        print("  ✅ All fields are default (as expected)")
                        all_passed = True
                    else:
                        print("  ❌ Expected all fields to be default")
                        all_passed = False

                status = "✅ PASS" if all_passed else "❌ FAIL"
                print(f"Overall: {status}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

        print("-" * 50)

    print("\n" + "=" * 60)
    print("TESTING COMPLEX NESTED MODELS")
    print("=" * 60)

    # Complex nested model test cases
    nested_test_cases: list[dict[str, Any]] = [
        # Case 1: Complete nested object
        {
            "input": """```json
            {
                "id": 123,
                "name": "John Doe",
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "Boston",
                    "country": "USA",
                    "postal_code": "02101"
                },
                "contact": {
                    "email": "john@example.com",
                    "phone": "+1-555-0123"
                },
                "is_premium": true
            }
            ```""",
            "description": "Complete nested object with address and contact",
            "model": ComplexUser,
            "expected_checks": {"name": "John Doe", "id": 123, "is_premium": True},
        },
        # Case 2: Partial nested object (streaming cutoff)
        {
            "input": '{"name": "Jane Smith", "address": {"street": "456 Oak Ave", "city": "Seattle"',
            "description": "Partial nested object - streaming cutoff mid-nested-object",
            "model": ComplexUser,
            "expected_checks": {"name": "Jane Smith"},
        },
        # Case 3: Missing nested object entirely but with flat fields
        {
            "input": '{"id": 456, "name": "Bob Wilson", "age": 45, "is_premium": false}',
            "description": "Flat object without nested fields",
            "model": ComplexUser,
            "expected_checks": {"name": "Bob Wilson", "id": 456, "is_premium": False},
        },
        # Case 4: Array of objects (Note: Pydantic doesn't automatically parse nested objects from JSON)
        {
            "input": """
            Looking great! Here's your project:
            {
                "project_name": "AI Assistant",
                "description": "Building an intelligent assistant",
                "budget": 150000.50,
                "completed": false
            }
            """,
            "description": "Project with basic fields (arrays of nested objects are complex)",
            "model": Project,
            "expected_checks": {
                "project_name": "AI Assistant",
                "budget": 150000.50,
                "completed": False,
            },
        },
        # Case 5: Deeply nested with some missing fields
        {
            "input": """
            {
                "name": "Alice Cooper",
                "contact": {
                    "email": "alice@company.com"
                },
                "company": {
                    "name": "Tech Corp",
                    "department": "Engineering"
                }
            }
            """,
            "description": "Nested objects with some missing fields",
            "model": ComplexUser,
            "expected_checks": {"name": "Alice Cooper"},
        },
        # Case 6: Malformed nested object
        {
            "input": '{"name": "Charlie", "address": {"street": "789 Pine", "city": "Denver"',
            "description": "Malformed nested object (missing closing braces)",
            "model": ComplexUser,
            "expected_checks": {"name": "Charlie"},
        },
        # Case 7: Empty nested objects
        {
            "input": '{"name": "Diana", "address": {}, "contact": {}, "age": 28}',
            "description": "Empty nested objects",
            "model": ComplexUser,
            "expected_checks": {"name": "Diana", "age": 28},
        },
        # Case 8: Mixed valid and invalid nested data
        {
            "input": """
            {
                "id": 789,
                "name": "Eve Anderson",
                "address": {
                    "street": "321 Elm St",
                    "city": "Portland",
                    "postal_code": "97201"
                },
                "contact": {
                    "email": "eve@test.com",
                    "phone": null
                },
                "metadata": {
                    "signup_date": "2023-01-15",
                    "source": "referral"
                }
            }
            """,
            "description": "Mixed nested data with metadata object",
            "model": ComplexUser,
            "expected_checks": {"name": "Eve Anderson", "id": 789},
        },
    ]

    # Run nested model tests
    for i, test_case in enumerate(nested_test_cases, 1):
        print(f"Nested Test {i}: {test_case['description']}")
        print(
            f"Input: {repr(test_case['input'][:100])}{'...' if len(test_case['input']) > 100 else ''}"
        )

        try:
            result = parse_streaming_json(test_case["input"], test_case["model"])
            print(f"Result: {result}")

            # Check expected values
            if "expected_checks" in test_case:
                all_passed = True
                for field, expected_value in test_case["expected_checks"].items():
                    actual_value = getattr(result, field, "MISSING_FIELD")
                    if actual_value == expected_value:
                        print(f"  ✅ {field}: {actual_value}")
                    else:
                        print(
                            f"  ❌ {field}: expected {expected_value}, got {actual_value}"
                        )
                        all_passed = False

                status = "✅ PASS" if all_passed else "❌ PARTIAL/FAIL"
                print(f"Overall: {status}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

        print("-" * 50)

    # Additional test with Profile model
    print("\n" + "=" * 40)
    print("ADDITIONAL TESTS")
    print("=" * 40)
    print("Testing with Profile model:")
    profile_input = """
    Looking good! Here's your profile data:
    {"user_id": 12345, "username": "coding_master", "bio": "I love Python!", "followers": 150, "verified": true
    """

    profile_result = parse_streaming_json(profile_input, Profile)
    print(f"Profile result: {profile_result}")

    # Test edge case - completely invalid input
    print("\nTesting edge case - completely invalid input:")
    invalid_result = parse_streaming_json("This has no JSON at all!", User)
    print(f"Invalid input result: {invalid_result}")

    # Test the fixed unquoted keys case
    print("\nRetesting unquoted keys (should now work):")
    unquoted_result = parse_streaming_json('{name: "Alice", age: 25}', User)
    print(f"Unquoted keys result: {unquoted_result}")
    expected_name = "Alice"
    actual_name = unquoted_result.name
    status = "✅ PASS" if actual_name == expected_name else "❌ FAIL"
    print(f"Expected name: {expected_name}, Got: {actual_name} - {status}")

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("Note: Complex nested objects require special handling in Pydantic.")
    print(
        "For full nested object support, consider using model_validate() with proper JSON parsing."
    )
