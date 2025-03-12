import PyPDF2
import difflib
import os
import sys
import re
import argparse
import json
import textwrap
from collections import defaultdict
import requests

# Try to import dotenv for environment variable management
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
    # Load environment variables from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    HAS_DOTENV = False
    print("Note: For better security, install python-dotenv: pip install python-dotenv")

# Try to import pdfplumber for better text extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Note: For better PDF text extraction, install pdfplumber: pip install pdfplumber")

# Try to import OpenAI for LLM analysis
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Note: For semantic analysis, install OpenAI: pip install openai")

def extract_text_from_pdf(pdf_path, verbose=False):
    """Extract text from a PDF file using multiple methods for better results."""
    text = ""
    
    # First try with pdfplumber if available (better text extraction)
    if HAS_PDFPLUMBER:
        try:
            if verbose:
                print(f"DEBUG: Attempting extraction with pdfplumber for {os.path.basename(pdf_path)}")
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=3)
                    if page_text:
                        text += page_text + "\n\n"  # Add double newline between pages
            
            if text and verbose:
                print(f"DEBUG: pdfplumber extracted {len(text.splitlines())} lines of text")
                
            if text:
                return text
        except Exception as e:
            if verbose:
                print(f"DEBUG: pdfplumber extraction failed: {str(e)}")
    
    # Fall back to PyPDF2 if pdfplumber failed or is not available
    try:
        if verbose:
            print(f"DEBUG: Attempting extraction with PyPDF2 for {os.path.basename(pdf_path)}")
            
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"  # Add double newline between pages
        
        if verbose:
            print(f"DEBUG: PyPDF2 extracted {len(text.splitlines())} lines of text")
    except Exception as e:
        print(f"Error reading {pdf_path} with PyPDF2: {e}")
        
    # If we still don't have text, report the error
    if not text:
        print(f"WARNING: Could not extract any text from {pdf_path}")
        if verbose:
            print("DEBUG: The PDF might be scanned or image-based. Consider using OCR software.")
    
    return text

def preprocess_text(text, verbose=False):
    """Advanced preprocessing to prepare text for comparison."""
    if verbose:
        original_line_count = len(text.splitlines())
        print(f"DEBUG: Preprocessing text with {original_line_count} original lines")
    
    # Replace tab characters with spaces
    text = text.replace('\t', ' ')
    
    # Fix common OCR/PDF extraction artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Remove non-printable chars
    
    # Make sure periods have spaces after them for better sentence detection
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # Insert line breaks at sentence boundaries if the text appears to be one long paragraph
    if len(text.splitlines()) < 5 and len(text) > 1000:
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)
        if verbose:
            print(f"DEBUG: Inserted line breaks at sentence boundaries, now has {len(text.splitlines())} lines")
    
    # Insert line breaks at potential section boundaries
    section_patterns = [
        r'(SECTION\s+\d+)', 
        r'(ARTICLE\s+\d+)', 
        r'(\d+\.\s+[A-Z][A-Za-z\s]+)',
        r'([A-Z][A-Z\s]{3,}:)'
    ]
    
    for pattern in section_patterns:
        text = re.sub(pattern, r'\n\n\1', text)
    
    # Normalize whitespace but preserve paragraphs
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\s*\n\s*', '\n', text)  # Clean up spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines with double newlines
    
    # Try to detect and preserve paragraph structure
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for para in paragraphs:
        # Handle bullets and numbering
        para = re.sub(r'(^|\n)(\d+\.\s*|\•\s*|\*\s*)', r'\1• ', para)
        processed_paragraphs.append(para)
    
    processed_text = '\n\n'.join(processed_paragraphs)
    
    # Make a final additional attempt to break long paragraphs into multiple lines
    if len(processed_text.splitlines()) < 10 and len(processed_text) > 1000:
        # Try to split by sentence boundaries with capital letters
        processed_text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', processed_text)
        
        # Try to split by numbered sections
        processed_text = re.sub(r'(\.\s+)(\d+\.)', r'\1\n\2', processed_text)
        
        if verbose:
            print(f"DEBUG: Made final attempt to split long text, now has {len(processed_text.splitlines())} lines")
    
    if verbose:
        processed_line_count = len(processed_text.splitlines())
        print(f"DEBUG: Text after preprocessing has {processed_line_count} lines")
        if processed_line_count < 10:
            print("DEBUG: WARNING - Text still has very few lines after preprocessing!")
            print("DEBUG: First 200 chars of text: " + processed_text[:200])
    
    return processed_text

def normalize_text(text, verbose=False):
    """Normalize text by removing extra whitespace and standardizing line endings."""
    # Apply advanced preprocessing first
    text = preprocess_text(text, verbose)
    
    # Split into lines and clean
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:  # Only keep non-empty lines
            lines.append(line)
    
    # Check if we have very few lines, which might indicate paragraphs weren't properly split
    if len(lines) < 10:
        if verbose:
            print(f"DEBUG: WARNING - Only {len(lines)} lines after normalization!")
            print(f"DEBUG: This might indicate the PDF text is not extracting correctly.")
            print(f"DEBUG: Attempting emergency text splitting to create more lines...")
        
        # For single line documents, try more aggressive splitting methods
        if len(lines) == 1 and len(lines[0]) > 500:
            # Try to split the text more aggressively
            new_lines = []
            current_line = lines[0]
            
            # Try to split by periods followed by capital letters
            split_text = re.split(r'(?<=[.!?])\s+(?=[A-Z])', current_line)
            if len(split_text) > 1:
                new_lines.extend(split_text)
            else:
                # Try to split by section titles
                split_text = re.split(r'(?:SECTION|ARTICLE)\s+\d+|(?:\d+\.)\s+[A-Z]|[A-Z][A-Z\s]{3,}:', current_line)
                if len(split_text) > 1:
                    # Reassemble with section headers
                    pattern = r'(?:(SECTION|ARTICLE)\s+\d+|(?:\d+\.)\s+[A-Z]|[A-Z][A-Z\s]{3,}:)'
                    headers = re.findall(pattern, current_line)
                    for i, segment in enumerate(split_text):
                        if i > 0 and i-1 < len(headers):
                            new_lines.append(headers[i-1] + segment)
                        else:
                            new_lines.append(segment)
                else:
                    # If all else fails, just split every few sentences
                    split_text = re.split(r'(?<=[.!?])\s+', current_line)
                    for i in range(0, len(split_text), 3):
                        chunk = ' '.join(split_text[i:i+3])
                        if chunk:
                            new_lines.append(chunk)
            
            if len(new_lines) > len(lines):
                if verbose:
                    print(f"DEBUG: Emergency splitting created {len(new_lines)} lines")
                lines = new_lines
    
    # Check one more time - if still very few lines, log a warning
    if len(lines) < 5 and verbose:
        print(f"DEBUG: SEVERE WARNING - Still only {len(lines)} lines after emergency splitting!")
        print(f"DEBUG: PDF text extraction may be failing. Manual review recommended.")
    
    return lines

def segment_contract_into_sections(lines, verbose=False):
    """Segment contract text into logical sections for better comparison."""
    # If there are too few lines, print a warning
    if len(lines) < 5:
        print(f"WARNING: Only {len(lines)} lines detected in contract. Text extraction may be incomplete.")
        if verbose:
            print("DEBUG: Contract text may need manual preprocessing or OCR.")
            for i, line in enumerate(lines):
                print(f"DEBUG: Line {i+1}: {line[:100]}...")

    # Common section header patterns in contracts
    section_patterns = [
        r'^(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+\d+',  # Section 1, Article II
        r'^(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+[IVX]+',  # Article IV
        r'^(?:\d+\.)\s+[A-Z][A-Za-z\s]+',  # 1. Definitions
        r'^[A-Z][A-Z\s]{3,}:',  # DEFINITIONS:
        r'^[A-Z][A-Z\s]{3,}$',   # TERMS AND CONDITIONS
        r'^\d+\.',   # Numbered sections (1.)
        r'^[A-Z]{1}[a-z]+\s+\d+:', # Paragraph 1:
        r'^[A-Z][a-z]+\s+\d+\.' # Paragraph 1.
    ]
    
    # Compile the patterns for efficiency
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in section_patterns]
    
    # Identify section boundaries
    section_indices = []
    section_titles = []
    
    for i, line in enumerate(lines):
        for pattern in compiled_patterns:
            if pattern.match(line):
                section_indices.append(i)
                section_titles.append(line[:100])  # Store the section title
                break
    
    # Make sure there's at least one section (the whole contract if no headers found)
    if not section_indices:
        if verbose:
            print("DEBUG: No clear section headers found, treating entire document as one section")
            
        # Last attempt to find sections in long text spans
        if len(lines) == 1 and len(lines[0]) > 1000:
            if verbose:
                print("DEBUG: Long single line detected, trying to identify sections within the text")
            for pattern in [r'(SECTION\s+\d+)', r'(ARTICLE\s+\d+)', r'(\d+\.\s+[A-Z][A-Za-z\s]+)']:
                if re.search(pattern, lines[0]):
                    if verbose:
                        print(f"DEBUG: Found potential section markers using pattern {pattern}")
                    # We found potential sections but they weren't properly split into lines
                    # This is a fallback message to help the user understand the issue
                    print("NOTE: Document contains sections but text extraction didn't separate them into lines.")
                    print("      Consider using --verbose mode for more details or try OCR if the PDF is scanned.")
                    break
        
        return [(0, len(lines), "ENTIRE CONTRACT")]
    
    # Add document end as the final boundary
    section_indices.append(len(lines))
    
    # Build sections as (start, end, title) tuples
    sections = []
    for i in range(len(section_indices) - 1):
        start = section_indices[i]
        end = section_indices[i + 1]
        title = section_titles[i] if i < len(section_titles) else "SECTION " + str(i+1)
        sections.append((start, end, title))
    
    if verbose:
        print(f"DEBUG: Identified {len(sections)} sections in the contract")
        for i, (_, _, title) in enumerate(sections):
            print(f"DEBUG:   Section {i+1}: {title[:50]}...")
    else:
        # Print section count even in non-verbose mode
        print(f"Identified {len(sections)} sections in the contract")
    
    return sections

def identify_changed_sections(reference_sections, new_lines, reference_lines, verbose=False):
    """Identify sections that have changes between the two contracts."""
    changed_sections = []
    
    # Keep track of matched sections for debugging
    matched_sections = {}
    processed_new_section_indices = set()
    
    for start, end, title in reference_sections:
        # Extract this section from the reference contract
        section_text = reference_lines[start:end]
        
        # Try to locate this section in the new contract using fuzzy matching
        # Normalize the title - remove DocuSign IDs and standardize case for better matching
        normalized_title = re.sub(r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', '', title, flags=re.IGNORECASE)
        normalized_title = normalized_title.strip()
        
        title_words = normalized_title.lower().split()
        
        best_match_idx = -1
        best_score = 0
        best_match_title = ""
        best_section_start = -1
        best_section_end = -1
        
        # Get the section text to use for content matching if needed
        original_section_text = ' '.join(reference_lines[start:end])
        
        # Look for matching section in the new contract
        new_sections = segment_contract_into_sections(new_lines, False)  # Don't need verbose output for this
        
        for new_start, new_end, new_title in new_sections:
            if (new_start, new_end) in processed_new_section_indices:
                continue  # Skip sections we've already matched
                
            # Normalize the line - remove DocuSign IDs for comparison
            normalized_new_title = re.sub(r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', '', new_title, flags=re.IGNORECASE)
            normalized_new_title = normalized_new_title.strip()
            new_title_lower = normalized_new_title.lower()
            
            # Calculate different similarity scores
            # 1. Title word overlap
            word_score = sum(1 for word in title_words if word in new_title_lower)
            
            # 2. Sequence similarity of titles
            title_similarity = difflib.SequenceMatcher(None, normalized_title.lower(), new_title_lower).ratio()
            
            # 3. Optional: Content similarity for sections with identical section numbers
            content_similarity = 0
            section_num_pattern = r'^\d+\.\d+(\.\d+)*'
            orig_section_num = re.match(section_num_pattern, normalized_title)
            new_section_num = re.match(section_num_pattern, normalized_new_title)
            
            if orig_section_num and new_section_num and orig_section_num.group(0) == new_section_num.group(0):
                # Same section numbers - boost score significantly
                content_similarity = 5  # Assign a high base score
            
            # Combine scores with weights
            total_score = word_score + (title_similarity * 10)
            if content_similarity > 0:
                total_score += content_similarity * 5  # Section numbers are strong indicators
            
            if total_score > best_score:
                best_score = total_score
                best_match_idx = new_start  # Use start index of the section
                best_match_title = new_title
                best_section_start = new_start
                best_section_end = new_end
        
        # If we found a potential match
        if best_match_idx >= 0 and best_score > 0:
            if verbose:
                print(f"DEBUG: Matched section '{title}' to '{best_match_title}' with score {best_score}")
                
            # Track this match for debugging
            matched_sections[title] = best_match_title
            processed_new_section_indices.add((best_section_start, best_section_end))
                
            # Extract the corresponding section from the new contract
            new_section_text = new_lines[best_section_start:best_section_end]
            
            # Compare the sections
            differ = difflib.Differ()
            diff = list(differ.compare(section_text, new_section_text))
            
            # Check if there are actual differences
            has_changes = any(line.startswith('- ') or line.startswith('+ ') for line in diff)
            
            if has_changes:
                # If this section has changes, add it to our list
                changed_sections.append((title, section_text, new_section_text))
                
                if verbose:
                    print(f"DEBUG: Found changes in section: {title}")
        else:
            if verbose:
                print(f"DEBUG: Could not find matching section for: {title}")
            # Add as a potentially removed section
            changed_sections.append((title, section_text, ["[SECTION NOT FOUND IN NEW CONTRACT]"]))
    
    # Look for new sections in the new contract
    new_sections = segment_contract_into_sections(new_lines, verbose)
    
    # If verbose, print all matched sections for debugging
    if verbose:
        print("\nDEBUG: Matched sections:")
        for orig, new in matched_sections.items():
            print(f"  '{orig}' → '{new}'")
    
    for new_start, new_end, new_title in new_sections:
        # Skip sections we've already processed
        if (new_start, new_end) in processed_new_section_indices:
            continue
            
        # Normalize the title for comparison
        normalized_new_title = re.sub(r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', '', new_title, flags=re.IGNORECASE).strip()
        
        # Check if this section title exists in any of the reference sections
        # Use a more fuzzy matching approach
        found_match = False
        for ref_title, _, _ in changed_sections:
            # Normalize the reference title too
            normalized_ref = re.sub(r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', '', ref_title, flags=re.IGNORECASE).strip()
            
            # Check for significant word overlap or sequence similarity
            title_words = set(normalized_new_title.lower().split())
            ref_words = set(normalized_ref.lower().split())
            
            # Skip very small word sets that might cause false matches
            if len(title_words) < 2 or len(ref_words) < 2:
                continue
                
            word_overlap = len(title_words.intersection(ref_words)) / max(len(title_words), len(ref_words)) if title_words else 0
            
            sequence_similarity = difflib.SequenceMatcher(None, normalized_new_title.lower(), normalized_ref.lower()).ratio()
            
            # Consider a match if either similarity metric is high enough
            if word_overlap > 0.5 or sequence_similarity > 0.6:
                found_match = True
                if verbose:
                    print(f"DEBUG: Section '{new_title}' matched to existing section '{ref_title}'")
                break
        
        if not found_match:
            # Check if this might be a DocuSign-only variation of an existing section
            docusign_id = re.search(r'DocuSign Envelope ID:\s*([A-F0-9\-]+)\s*', new_title, flags=re.IGNORECASE)
            if docusign_id:
                # Extract the part after the DocuSign ID
                rest_of_title = re.sub(r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', '', new_title, flags=re.IGNORECASE).strip()
                
                # Check if this remaining text matches any existing sections
                for ref_title, _, _ in changed_sections:
                    normalized_ref = re.sub(r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', '', ref_title, flags=re.IGNORECASE).strip()
                    if rest_of_title and normalized_ref and (
                            rest_of_title.lower() in normalized_ref.lower() or 
                            normalized_ref.lower() in rest_of_title.lower() or
                            difflib.SequenceMatcher(None, rest_of_title.lower(), normalized_ref.lower()).ratio() > 0.6):
                        found_match = True
                        if verbose:
                            print(f"DEBUG: DocuSign section '{new_title}' matched to existing section '{ref_title}'")
                        break
            
            if not found_match:
                # This might be a new section
                new_section_text = new_lines[new_start:new_end]
                changed_sections.append((f"NEW SECTION: {new_title}", ["[NEW SECTION ADDED TO CONTRACT]"], new_section_text))
                
                if verbose:
                    print(f"DEBUG: Found new section: {new_title}")
    
    return changed_sections

def get_api_key(verbose=False):
    """Get the OpenAI API key from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        # Check alternative environment variable names
        api_key = os.environ.get("OPENAI_API") or os.environ.get("OPENAI_KEY")
        
    if not api_key and verbose:
        print("DEBUG: No OpenAI API key found in environment variables")
    
    return api_key

def analyze_section_with_llm(section_title, original_text, revised_text, verbose=False):
    """Use an LLM to analyze differences between original and revised contract sections.
    
    Args:
        section_title (str): The title of the section being analyzed
        original_text (str): The text of the section in the original contract
        revised_text (str): The text of the section in the revised contract
        verbose (bool): Whether to output verbose debugging information
    
    Returns:
        dict: Analysis results including changes, impact, legal implications, and significance
    """
    if not revised_text:
        # Section was removed
        return {
            "changes": "This section was completely removed in the revised contract.",
            "impact": "The removal of this section may significantly affect rights or obligations previously established.",
            "legal_implications": "The removal of contract language can have legal implications by eliminating previously established rights, obligations, or protections.",
            "significance": 5  # High significance for removed sections
        }
    
    if not original_text or original_text == ["[NEW SECTION ADDED TO CONTRACT]"]:
        # New section added
        return {
            "changes": "This section was newly added to the revised contract.",
            "impact": "This new section introduces rights or obligations not present in the original contract.",
            "legal_implications": "New contract language can create new legal requirements, rights, or protections not previously established.",
            "significance": 4  # High significance for new sections
        }

    # Preprocess text to handle common cases before detailed analysis
    # Handle metadata changes (like DocuSign IDs, page numbers, etc.) separately
    metadata_patterns = [
        (r'DocuSign Envelope ID:\s*[A-F0-9\-]+\s*', 'DocuSign Envelope ID'),
        (r'Page \d+ of \d+', 'Page numbering'),
        (r'^\s*Date:\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$', 'Date field')
    ]
    
    # Check for metadata differences
    metadata_changes = []
    for pattern, desc in metadata_patterns:
        original_matches = re.finditer(pattern, original_text, re.IGNORECASE | re.MULTILINE)
        revised_matches = re.finditer(pattern, revised_text, re.IGNORECASE | re.MULTILINE)
        
        orig_values = [m.group(0) for m in original_matches]
        rev_values = [m.group(0) for m in revised_matches]
        
        # Check for additions, removals, or changes in these metadata elements
        if orig_values != rev_values:
            if not orig_values and rev_values:
                metadata_changes.append(f"{desc} added: {', '.join(rev_values)}")
            elif orig_values and not rev_values:
                metadata_changes.append(f"{desc} removed: {', '.join(orig_values)}")
            else:
                metadata_changes.append(f"{desc} changed from '{', '.join(orig_values)}' to '{', '.join(rev_values)}'")
    
    # Clean text for more focused comparison
    original_cleaned = original_text
    revised_cleaned = revised_text
    
    for pattern, _ in metadata_patterns:
        original_cleaned = re.sub(pattern, '', original_cleaned, flags=re.IGNORECASE | re.MULTILINE)
        revised_cleaned = re.sub(pattern, '', revised_cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # If after removing metadata the texts are identical, return low significance result
    if metadata_changes and original_cleaned.strip() == revised_cleaned.strip():
        changes_desc = '; '.join(metadata_changes)
        return {
            "changes": f"Only metadata was changed in this section: {changes_desc}",
            "impact": "These changes are administrative and do not affect the substantive rights or obligations of the parties.",
            "legal_implications": "No significant legal implications as these are procedural/administrative changes.",
            "significance": 1  # Low significance for metadata-only changes
        }
    
    # Check for numerical changes (dates, amounts, time periods)
    num_pattern = r'(\$\s*[\d,]+(\.\d+)?|\d+\s*(st|nd|rd|th|days?|months?|years?|weeks?|dollars?|cents?)|\d{1,2}/\d{1,2}/\d{2,4})'
    
    original_nums = re.findall(num_pattern, original_text, re.IGNORECASE)
    revised_nums = re.findall(num_pattern, revised_text, re.IGNORECASE)
    
    num_changes = []
    # Extract just the matched strings
    original_num_values = [match[0] for match in original_nums]
    revised_num_values = [match[0] for match in revised_nums]
    
    # Find numbers that changed - this is a simple implementation
    # For production, you'd want a more sophisticated approach to match corresponding numbers
    if len(original_num_values) == len(revised_num_values):
        for i, (orig, rev) in enumerate(zip(original_num_values, revised_num_values)):
            if orig != rev:
                num_changes.append(f"Changed from '{orig}' to '{rev}'")
    
    # If only numeric changes detected, we can include this info in the LLM prompt
    numeric_change_info = ""
    if num_changes:
        numeric_change_info = "Note these potential numeric changes detected: " + "; ".join(num_changes)
    
    # If texts are completely identical, return immediately
    if original_text.strip() == revised_text.strip():
        return {
            "changes": "No changes were made to this section. The text is identical in both versions.",
            "impact": "No impact as the section remains unchanged.",
            "legal_implications": "No legal implications as the section remains unchanged.",
            "significance": 1  # Low significance for unchanged sections
        }
    
    # Use LLM to analyze differences
    api_key = get_api_key(verbose)
    
    # Prepare the prompt for the LLM
    prompt = f"""
You are a legal expert specializing in contract analysis. Your task is to compare two versions of the same contract section and identify meaningful differences, especially focusing on:

1. Dollar amounts
2. Time requirements/deadlines
3. Substantive terms that change rights or obligations

{numeric_change_info}

Pay special attention to numerical changes (like "120th day" to "180th day") and ignore administrative changes like the addition of document IDs, page numbers, or formatting unless they have legal significance.

Section Title: {section_title}

ORIGINAL TEXT:
```
{original_text}
```

REVISED TEXT:
```
{revised_text}
```

Please provide:

1. EXACT TEXT CHANGES: Quote the specific text that differs between versions. Show "Original: [text]" and "Revised: [text]" for each change.

2. SUMMARY OF CHANGES: Describe what has changed in simple terms, focusing on substantive differences. Pay special attention to numerical values, dates, deadlines, and other specific terms.

3. PRACTICAL IMPACT: Explain how these changes affect what the parties must do, can do, or are responsible for. Focus on real-world consequences.

4. LEGAL IMPLICATIONS: Briefly describe how these changes might affect legal rights or responsibilities.

5. SIGNIFICANCE RATING: Rate how significant these changes are on a scale of 1-5:
   - 1: No substantive change (formatting only)
   - 2: Minor change with minimal practical impact
   - 3: Moderate change affecting some rights or obligations
   - 4: Significant change with substantial impact
   - 5: Critical change fundamentally altering key terms

If there are no substantive changes (just formatting, spelling corrections, document IDs, etc.), clearly state this and rate significance as 1.
"""
    
    if verbose:
        print(f"DEBUG: Sending prompt to LLM for section: {section_title}")
        if numeric_change_info:
            print(f"DEBUG: Included numeric change info: {numeric_change_info}")
    
    # Call the LLM API
    headers = {"Authorization": f"Bearer {api_key}"}
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a legal expert specializing in contract analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API request failed: {e}")
        return {
            "changes": "Error analyzing section with LLM.",
            "impact": "Unable to determine impact due to API error.",
            "legal_implications": "Unable to analyze legal implications due to API error.",
            "significance": 3  # Default to medium significance when we can't analyze
        }
    
    # Process the response
    try:
        analysis = response.json()["choices"][0]["message"]["content"]
        
        if verbose:
            print(f"DEBUG: Received analysis from LLM for section: {section_title}")
            print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
        
        # Extract information from the LLM response
        changes = ""
        impact = ""
        legal_implications = ""
        significance = 1
        
        # Look for sections in the response
        if "EXACT TEXT CHANGES:" in analysis:
            changes_section = analysis.split("EXACT TEXT CHANGES:")[1].split("SUMMARY OF CHANGES:" if "SUMMARY OF CHANGES:" in analysis else "PRACTICAL IMPACT:")[0].strip()
            changes = changes_section
        
        if "SUMMARY OF CHANGES:" in analysis:
            summary_section = analysis.split("SUMMARY OF CHANGES:")[1].split("PRACTICAL IMPACT:" if "PRACTICAL IMPACT:" in analysis else "LEGAL IMPLICATIONS:")[0].strip()
            if changes:
                changes += "\n\n" + summary_section
            else:
                changes = summary_section
        
        if "PRACTICAL IMPACT:" in analysis:
            practical_section = analysis.split("PRACTICAL IMPACT:")[1].split("LEGAL IMPLICATIONS:" if "LEGAL IMPLICATIONS:" in analysis else "SIGNIFICANCE RATING:")[0].strip()
            impact = practical_section
        
        if "LEGAL IMPLICATIONS:" in analysis:
            legal_section = analysis.split("LEGAL IMPLICATIONS:")[1].split("SIGNIFICANCE RATING:" if "SIGNIFICANCE RATING:" in analysis else "")[0].strip()
            legal_implications = legal_section
        
        if "SIGNIFICANCE RATING:" in analysis:
            rating_section = analysis.split("SIGNIFICANCE RATING:")[1].strip()
            # Find first digit in the rating section
            for char in rating_section:
                if char.isdigit():
                    significance = int(char)
                    break
        
        # If we couldn't extract structured information, handle accordingly
        if not changes:
            if "no substantive changes" in analysis.lower() or "no changes" in analysis.lower():
                changes = "No substantive changes were identified between the two versions."
                significance = 1
            else:
                changes = analysis
        
        if not impact:
            impact = "Impact not specifically identified in the analysis."
        
        if not legal_implications:
            legal_implications = "Legal implications not specifically identified in the analysis."
        
        return {
            "changes": changes,
            "impact": impact,
            "legal_implications": legal_implications,
            "significance": significance
        }
    
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to process LLM response: {e}")
        return {
            "changes": "Error processing LLM analysis.",
            "impact": "Unable to determine impact due to processing error.",
            "legal_implications": "Unable to analyze legal implications due to processing error.",
            "significance": 3  # Default to medium significance when we can't analyze
        }

def compare_contracts_with_llm(reference_text, new_text, reference_path, comparison_path, verbose=False):
    """Compare contract texts using LLM to identify and analyze differences.
    
    Args:
        reference_text (str): The reference (original) contract text
        new_text (str): The new (revised) contract text
        reference_path (str): Path to reference contract file
        comparison_path (str): Path to new contract file
        verbose (bool): Whether to output verbose debugging information
    """
    # Parse the contracts into lines, sections, etc.
    reference_lines = preprocess_text(reference_text, verbose).split('\n')
    new_lines = preprocess_text(new_text, verbose).split('\n')
    
    print(f"\nOriginal contract has {len(reference_lines)} lines")
    print(f"Revised contract has {len(new_lines)} lines\n")
    
    # Segment contracts into sections
    print("Identifying contract sections...")
    reference_sections = segment_contract_into_sections(reference_lines, verbose)
    new_sections = segment_contract_into_sections(new_lines, verbose)
    
    print(f"Identified {len(reference_sections)} sections in the original contract")
    print(f"Identified {len(new_sections)} sections in the revised contract\n")
    
    # Create a mapping of section titles to their content for both contracts
    ref_section_map = {}
    for start, end, title in reference_sections:
        section_text = '\n'.join(reference_lines[start:end])
        ref_section_map[title] = {"text": section_text, "position": (start, end)}
    
    new_section_map = {}
    for start, end, title in new_sections:
        section_text = '\n'.join(new_lines[start:end])
        new_section_map[title] = {"text": section_text, "position": (start, end)}
    
    # Initialize tracking for sections in revised but not in original
    sections_only_in_revised = set(new_section_map.keys())
    
    # Track results
    analysis_results = []
    
    # Generate a unique output filename based on the contract names
    reference_basename = os.path.basename(reference_path)
    comparison_basename = os.path.basename(comparison_path)
    
    # Get just the identifying part of the filename (e.g., "Lot_A", "Lot_B")
    reference_id = reference_basename.split("_")[-1].split(".")[0]
    comparison_id = comparison_basename.split("_")[-1].split(".")[0]
    
    output_file = f"contract_comparison_Lot_{reference_id}_vs_Lot_{comparison_id}.txt"
    
    with open(output_file, "w") as f:
        f.write("CONTRACT COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Original Contract: {reference_path}\n")
        f.write(f"Revised Contract: {comparison_path}\n\n")
        f.write("=" * 80 + "\n\n")
    
    # Process all sections in the original contract
    total_sections = len(reference_sections)
    print(f"Analyzing all {total_sections} contract sections...")
    
    # Create a progress bar for visual feedback
    progress_bar_width = 60
    
    for i, (start, end, title) in enumerate(reference_sections):
        # Update progress bar
        percent_complete = (i / total_sections)
        filled_width = int(progress_bar_width * percent_complete)
        bar = '█' * filled_width + '░' * (progress_bar_width - filled_width)
        print(f"\rAnalyzing section {i+1}/{total_sections}: {title[:50]}...", end='')
        print(f"\rProgress: [{bar}] {i}/{total_sections} sections", end='')
        
        # Get original section text
        original_text = '\n'.join(reference_lines[start:end])
        
        # Find the matching section in the revised contract
        matching_section = None
        matching_text = None
        
        # Check for exact title match first
        if title in new_section_map:
            matching_section = title
            matching_text = new_section_map[title]["text"]
            sections_only_in_revised.discard(title)
        else:
            # Try to find the closest matching section title
            best_similarity = 0
            for new_title in new_section_map.keys():
                # Calculate similarity between titles
                similarity = difflib.SequenceMatcher(None, title.lower(), new_title.lower()).ratio()
                if similarity > best_similarity and similarity > 0.8:  # Threshold for considering a match
                    best_similarity = similarity
                    matching_section = new_title
                    matching_text = new_section_map[new_title]["text"]
            
            if matching_section:
                sections_only_in_revised.discard(matching_section)
        
        if matching_text is None:
            # Section was removed in the revised contract
            with open(output_file, "a") as f:
                f.write(f"SECTION REMOVED: {title}\n")
                f.write("-" * 80 + "\n")
                f.write("This section exists in the original contract but was removed in the revised version.\n")
                f.write("Original text:\n")
                f.write(original_text + "\n\n")
                f.write("-" * 80 + "\n\n")
            
            analysis_results.append({
                "section": title,
                "changes": "Section was completely removed in the revised contract.",
                "significance": 5,  # High significance for removed sections
                "impact": "This section was removed entirely, which may significantly change rights or obligations.",
                "original_text": original_text,
                "revised_text": None
            })
        else:
            # Both versions have this section, analyze the differences
            result = analyze_section_with_llm(title, original_text, matching_text, verbose)
            
            # Check if there are meaningful differences
            if result["changes"] and "no changes" not in result["changes"].lower():
                # Save analysis to file
                with open(output_file, "a") as f:
                    f.write(f"SECTION: {title}\n")
                    f.write("-" * 80 + "\n")
                    f.write("ORIGINAL TEXT:\n")
                    f.write(original_text + "\n\n")
                    f.write("REVISED TEXT:\n")
                    f.write(matching_text + "\n\n")
                    f.write("ANALYSIS:\n")
                    f.write(f"Changes: {result['changes']}\n")
                    f.write(f"Impact: {result['impact']}\n")
                    f.write(f"Legal Implications: {result['legal_implications']}\n")
                    f.write(f"Significance (1-5): {result['significance']}\n")
                    f.write("-" * 80 + "\n\n")
                
                analysis_results.append({
                    "section": title,
                    "changes": result["changes"],
                    "significance": result["significance"],
                    "impact": result["impact"],
                    "legal_implications": result["legal_implications"],
                    "original_text": original_text,
                    "revised_text": matching_text
                })
    
    # Process sections that exist only in the revised contract
    for title in sections_only_in_revised:
        print(f"\rAnalyzing new section: {title[:50]}...", end='')
        new_section_text = new_section_map[title]["text"]
        
        # Save analysis to file
        with open(output_file, "a") as f:
            f.write(f"NEW SECTION: {title}\n")
            f.write("-" * 80 + "\n")
            f.write("This section was added in the revised contract and does not exist in the original version.\n")
            f.write("New text:\n")
            f.write(new_section_text + "\n\n")
            f.write("-" * 80 + "\n\n")
        
        # Use LLM to analyze the significance of the new section
        result = analyze_section_with_llm("NEW SECTION: " + title, "", new_section_text, verbose)
        
        analysis_results.append({
            "section": "NEW SECTION: " + title,
            "changes": "This section was added in the revised contract.",
            "significance": result["significance"],
            "impact": result["impact"],
            "legal_implications": result["legal_implications"],
            "original_text": None,
            "revised_text": new_section_text
        })
    
    # Complete progress bar
    filled_width = progress_bar_width
    bar = '█' * filled_width + '░' * (progress_bar_width - filled_width)
    print(f"\rProgress: [{bar}] {total_sections}/{total_sections} sections - Complete!")
    
    # Print a blank line after the progress bar
    print("\n")
    print("=" * 80)
    print("SEMANTIC ANALYSIS RESULTS:")
    print("=" * 80)
    print()
    
    # Count meaningful changes
    meaningful_changes = [r for r in analysis_results if r["significance"] > 1]
    print(f"Showing {len(meaningful_changes)} sections with meaningful changes out of {total_sections} total sections")
    print("Use --verbose mode to see all section analyses")
    print("\n")
    
    # Sort analysis results by significance (highest first)
    analysis_results.sort(key=lambda x: x["significance"], reverse=True)
    
    # Display the top most significant changes
    for i, result in enumerate(analysis_results[:10], 1):
        if result["significance"] > 1:  # Only show significant changes
            print(f"{i}. SECTION: {result['section']}")
            print("-" * 80)
            
            # Display exact text changes if available
            if result["original_text"] and result["revised_text"]:
                print("ORIGINAL TEXT:")
                print(textwrap.fill(result["original_text"][:500], 80))
                print("\nREVISED TEXT:")
                print(textwrap.fill(result["revised_text"][:500], 80))
                print()
            
            print(result["changes"])
            print(f"Impact on Rights and Obligations: {result['impact']}")
            print(f"Legal Implications: {result['legal_implications']}")
            print(f"Significance Rating: {result['significance']}/5")
            print("-" * 80)
            print()
    
    # Generate executive summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY - WHAT YOU SHOULD CARE ABOUT")
    print("=" * 80)
    print()
    
    exec_summary = generate_executive_summary(analysis_results, verbose)
    print(exec_summary)
    
    # Append executive summary to the output file
    with open(output_file, "a") as f:
        f.write("\nEXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(exec_summary)
    
    print("=" * 80)
    print(f"ANALYSIS COMPLETE - Full results saved to {output_file}")
    print("=" * 80)
    
    return analysis_results

def is_meaningful_difference(text1, text2, verbose=False):
    """Determine if a difference is meaningful (not just formatting or minor change)."""
    # Remove all whitespace for comparison
    clean1 = re.sub(r'\s+', '', text1.lower())
    clean2 = re.sub(r'\s+', '', text2.lower())
    
    # If they're identical after whitespace removal, it's not meaningful
    if clean1 == clean2:
        if verbose:
            print(f"DEBUG: Texts identical after whitespace removal")
        return False, 1.0
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, clean1, clean2).ratio()
    
    # If very similar (over 95% match), it might just be minor formatting
    if similarity > 0.95:
        if verbose:
            print(f"DEBUG: Similarity too high ({similarity:.2f})")
        return False, similarity
    
    # Ignore differences that are just in punctuation or capitalization
    alpha_only1 = re.sub(r'[^a-z0-9]', '', clean1)
    alpha_only2 = re.sub(r'[^a-z0-9]', '', clean2)
    if alpha_only1 == alpha_only2:
        if verbose:
            print(f"DEBUG: Only punctuation/capitalization differences")
        return False, similarity
    
    if verbose:
        print(f"DEBUG: Meaningful difference found (similarity: {similarity:.2f})")
        print(f"  ORIGINAL: {text1}")
        print(f"  REVISED:  {text2}")
    
    return True, similarity

def find_section_context(lines, idx, context_lines=3):
    """Find surrounding context to identify which section of the contract we're in."""
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    
    # Look for section headings in the context
    section_pattern = re.compile(r'^(SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+\d+', re.IGNORECASE)
    heading_pattern = re.compile(r'^[A-Z\s]{5,}:')
    
    context = []
    section_heading = None
    
    for i in range(start, end):
        if i == idx:
            continue  # Skip the actual difference line
        line = lines[i]
        if section_pattern.search(line) or heading_pattern.search(line):
            section_heading = line
            break
        context.append(line)
    
    if section_heading:
        return section_heading
    
    # If no clear section heading, try to get context from surrounding lines
    context_text = " ".join(context).strip()
    if len(context_text) > 70:
        context_text = context_text[:70] + "..."
    
    return context_text if context_text else "Unknown section"

def summarize_changes(changes):
    """Create a summary of the meaningful changes found."""
    if not changes:
        return "No meaningful differences found between contracts."
    
    summary = []
    summary.append(f"Found {len(changes)} meaningful differences in contract language:")
    
    for i, (section, old, new, _) in enumerate(changes, 1):
        summary.append(f"{i}. In section related to: \"{section}\"")
    
    return "\n".join(summary)

def compare_texts(reference_text, new_text, reference_path, comparison_path, verbose=False):
    """Compare two texts and highlight meaningful differences in contract language."""
    # Get just the filename for display
    reference_filename = os.path.basename(reference_path)
    comparison_filename = os.path.basename(comparison_path)
    
    # Generate a unique output filename based on the contract names
    reference_basename = os.path.basename(reference_path)
    comparison_basename = os.path.basename(comparison_path)
    
    # Get just the identifying part of the filename (e.g., "Lot_A", "Lot_B")
    reference_id = reference_basename.split("_")[-1].split(".")[0]
    comparison_id = comparison_basename.split("_")[-1].split(".")[0]
    
    output_file = f"contract_comparison_Lot_{reference_id}_vs_Lot_{comparison_id}.txt"
    
    # Print header with file information
    print("\n" + "="*80)
    print(f"CONTRACT COMPARISON:")
    print(f"ORIGINAL: {reference_filename}")
    print(f"REVISED:  {comparison_filename}")
    print("="*80)
    
    # Normalize both texts to ignore whitespace differences
    reference_lines = normalize_text(reference_text, verbose)
    new_lines = normalize_text(new_text, verbose)
    
    if verbose:
        print(f"\nDEBUG: Original contract has {len(reference_lines)} lines")
        print(f"DEBUG: Revised contract has {len(new_lines)} lines")
        
        # Diagnostic info if very few lines
        if len(reference_lines) < 5 or len(new_lines) < 5:
            print("\nDEBUG: WARNING - Very few lines extracted. PDF text extraction may have failed.")
            print("DEBUG: Consider trying a different PDF library or OCR tool.")
            if len(reference_lines) > 0:
                print(f"\nDEBUG: First line from original: {reference_lines[0][:100]}...")
            if len(new_lines) > 0:
                print(f"\nDEBUG: First line from revised: {new_lines[0][:100]}...")
    
    # Use SequenceMatcher for better matching of contract language
    matcher = difflib.SequenceMatcher(None, reference_lines, new_lines)
    
    # Track meaningful differences
    meaningful_changes = []
    
    # For verbose mode: keep track of all sections seen
    if verbose:
        all_sections = set()
        total_comparisons = 0
        total_changes = 0
        similarity_sum = 0
    
    # Process the differences
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if verbose and tag != 'equal':
            print(f"\nDEBUG: Found {tag} operation from line {i1}-{i2} to {j1}-{j2}")
        
        if tag in ('replace', 'delete', 'insert'):
            # For each change, examine whether it's meaningful
            if tag == 'replace':
                for idx in range(i1, i2):
                    for jdx in range(j1, j2):
                        old_text = reference_lines[idx]
                        new_text = new_lines[jdx]
                        
                        if verbose:
                            total_comparisons += 1
                            section_context = find_section_context(reference_lines, idx)
                            all_sections.add(section_context)
                            print(f"\nDEBUG: Comparing texts in section: \"{section_context}\"")
                        
                        meaningful, similarity = is_meaningful_difference(old_text, new_text, verbose)
                        
                        if verbose:
                            similarity_sum += similarity
                        
                        if meaningful:
                            if verbose:
                                total_changes += 1
                            # Find which section this is in
                            section_context = find_section_context(reference_lines, idx)
                            meaningful_changes.append((section_context, old_text, new_text, similarity))
                            break
            elif tag == 'delete':
                # Something was removed from the original contract
                for idx in range(i1, i2):
                    old_text = reference_lines[idx]
                    section_context = find_section_context(reference_lines, idx)
                    
                    if verbose:
                        total_changes += 1
                        all_sections.add(section_context)
                        print(f"\nDEBUG: Text removed in section: \"{section_context}\"")
                        print(f"  REMOVED: {old_text}")
                    
                    meaningful_changes.append((section_context, old_text, "REMOVED", 0.0))
            elif tag == 'insert':
                # Something was added to the new contract
                for jdx in range(j1, j2):
                    new_text = new_lines[jdx]
                    # Try to find context in nearby unchanged sections
                    prev_i = i1 - 1 if i1 > 0 else 0
                    section_context = find_section_context(reference_lines, prev_i)
                    
                    if verbose:
                        total_changes += 1
                        all_sections.add(section_context)
                        print(f"\nDEBUG: Text added in section: \"{section_context}\"")
                        print(f"  ADDED: {new_text}")
                    
                    meaningful_changes.append((section_context, "ADDED", new_text, 0.0))
    
    # Print verbose statistics if requested
    if verbose and total_comparisons > 0:
        print("\n" + "="*80)
        print(f"VERBOSE VALIDATION SUMMARY:")
        print(f"Total sections examined: {len(all_sections)}")
        print(f"Total text comparisons made: {total_comparisons}")
        print(f"Average similarity score: {similarity_sum / total_comparisons:.2f}" if total_comparisons > 0 else "No comparisons made")
        print(f"Total changes detected: {total_changes}")
        print(f"Meaningful changes reported: {len(meaningful_changes)}")
        
        print("\nALL SECTIONS EXAMINED:")
        for i, section in enumerate(sorted(all_sections), 1):
            print(f"  {i}. {section}")
        print("="*80)
    
    # Display a summary first
    summary = summarize_changes(meaningful_changes)
    print(f"\n{summary}")
    
    # Create and write results to the output file
    with open(output_file, "w") as f:
        f.write("CONTRACT COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Original Contract: {reference_path}\n")
        f.write(f"Revised Contract: {comparison_path}\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Write summary
        f.write(f"{summary}\n\n")
    
    # If we found meaningful changes, display them in detail
    if meaningful_changes:
        print("\n" + "-"*80)
        print("DETAILED CONTRACT LANGUAGE CHANGES:")
        print("-"*80)
        
        # Group by section for more organized display
        sections = defaultdict(list)
        for section, old, new, similarity in meaningful_changes:
            sections[section].append((old, new, similarity))
        
        # Also write detailed changes to the output file
        with open(output_file, "a") as f:
            f.write("\n" + "-"*80 + "\n")
            f.write("DETAILED CONTRACT LANGUAGE CHANGES:\n")
            f.write("-"*80 + "\n\n")
        
        for i, (section, changes) in enumerate(sections.items(), 1):
            print(f"\n{i}. IN SECTION: \"{section}\"")
            print("   " + "-"*70)
            
            # Add to output file
            with open(output_file, "a") as f:
                f.write(f"\n{i}. IN SECTION: \"{section}\"\n")
                f.write("   " + "-"*70 + "\n")
            
            for old, new, similarity in changes:
                if verbose:
                    similarity_info = f" [similarity: {similarity:.2f}]" if similarity > 0 else ""
                else:
                    similarity_info = ""
                    
                if old == "ADDED":
                    detail = f"   [ADDED TO {comparison_filename}]{similarity_info}\n   + {new}"
                    print(detail)
                    with open(output_file, "a") as f:
                        f.write(detail + "\n")
                elif new == "REMOVED":
                    detail = f"   [REMOVED FROM {comparison_filename}]{similarity_info}\n   - {old}"
                    print(detail)
                    with open(output_file, "a") as f:
                        f.write(detail + "\n")
                else:
                    detail = f"   [IN {reference_filename}]{similarity_info}\n   - {old}\n   [IN {comparison_filename}]\n   + {new}"
                    print(detail)
                    with open(output_file, "a") as f:
                        f.write(detail + "\n")
                
                print("   " + "-"*70)
                with open(output_file, "a") as f:
                    f.write("   " + "-"*70 + "\n")
    
    print("="*80)
    print(f"ANALYSIS COMPLETE - Full results saved to {output_file}")
    print("="*80 + "\n")

def generate_executive_summary(analysis_results, verbose=False):
    """Generate an executive summary of the most important changes in the contract.
    
    Args:
        analysis_results (list): List of dictionaries containing analysis results
        verbose (bool): Whether to output verbose debugging information
    
    Returns:
        str: Executive summary text
    """
    # Sort analysis results by significance (highest first)
    sorted_results = sorted(analysis_results, key=lambda x: x["significance"], reverse=True)
    
    # Begin constructing the summary
    summary = "KEY CONTRACT CHANGES YOU SHOULD KNOW ABOUT:\n\n"
    
    # Add top significant changes
    high_priority_sections = []
    medium_priority_sections = []
    low_priority_sections = []
    
    for i, result in enumerate(sorted_results, 1):
        significance = result["significance"]
        section = result["section"]
        changes = result["changes"]
        impact = result["impact"]
        
        # Skip truly insignificant changes
        if significance == 1:
            low_priority_sections.append((section, "No substantive changes"))
            continue
        
        # Format section details
        section_summary = f"{i}. {section}\n"
        section_summary += f"   Significance: {significance}/5\n"
        
        # Extract exact text changes if available
        if "Original:" in changes and "Revised:" in changes:
            original_pattern = r'Original:\s*["\'](.*?)["\']'
            revised_pattern = r'Revised:\s*["\'](.*?)["\']'
            
            original_matches = re.findall(original_pattern, changes)
            revised_matches = re.findall(revised_pattern, changes)
            
            if original_matches and revised_matches:
                section_summary += "   EXACT TEXT CHANGES:\n"
                section_summary += f'   Original: "{original_matches[0]}"\n'
                section_summary += f'   Revised:  "{revised_matches[0]}"\n\n'
        
        section_summary += "   SUMMARY OF CHANGES:\n"
        section_summary += "   • " + changes.replace("\n", "\n   ") + "\n\n"
        
        section_summary += "   PRACTICAL IMPACT:\n"
        section_summary += "   • " + impact.replace("\n", "\n   ") + "\n\n"
        
        summary += section_summary
        
        # Categorize for priority list
        if significance >= 4:
            reason = "Important contractual changes"
            if "time" in changes.lower() or "day" in changes.lower():
                reason = f"Changed time requirements"
            elif "dollar" in changes.lower() or "$" in changes:
                reason = f"Changed financial terms"
            high_priority_sections.append((section, reason))
        elif significance >= 2:
            medium_priority_sections.append((section, "Moderate changes to terms"))
    
    # Add a section divider
    summary += "-" * 80 + "\n"
    summary += "SECTIONS YOU SHOULD RE-READ:\n"
    summary += "-" * 80 + "\n\n"
    
    # High priority sections
    if high_priority_sections:
        summary += "HIGH PRIORITY (Critical to review):\n"
        for i, (section, reason) in enumerate(high_priority_sections, 1):
            summary += f"  {i}. {section}\n"
            summary += f"     Why: {reason}\n"
        summary += "\n"
    
    # Medium priority sections
    if medium_priority_sections:
        summary += "MEDIUM PRIORITY:\n"
        for i, (section, reason) in enumerate(medium_priority_sections, 1):
            summary += f"  {i}. {section}\n"
        summary += "\n"
    
    # If no high or medium priority sections were found
    if not high_priority_sections and not medium_priority_sections:
        summary += "No sections requiring immediate attention were identified.\n"
    
    return summary

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Compare contract language in PDF files')
    parser.add_argument('reference_pdf', help='Path to the reference/original PDF contract')
    parser.add_argument('comparison', nargs='?', help='Path to comparison PDF file or directory (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode for validation')
    parser.add_argument('--llm', action='store_true', help='Use LLM for semantic analysis of differences')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if the reference file exists
    reference_pdf = args.reference_pdf
    if not os.path.exists(reference_pdf):
        print(f"Reference file {reference_pdf} not found.")
        return
    
    # Get absolute path for better readability in output
    reference_pdf_abs = os.path.abspath(reference_pdf)
    
    # Extract text from the reference contract
    print(f"Reading original contract: {os.path.basename(reference_pdf_abs)}")
    reference_text = extract_text_from_pdf(reference_pdf_abs, args.verbose)
    if not reference_text:
        print("Could not extract text from reference contract.")
        return
    
    # If a comparison argument is provided, use it as the new PDF or directory
    if args.comparison:
        new_input = args.comparison
        new_input_abs = os.path.abspath(new_input)
        
        # Check if the input is a file
        if os.path.isfile(new_input) and new_input.endswith(".pdf"):
            print(f"Reading revised contract: {os.path.basename(new_input_abs)}")
            new_text = extract_text_from_pdf(new_input_abs, args.verbose)
            if new_text:
                if args.llm:
                    # Use LLM-enhanced comparison
                    if HAS_OPENAI:
                        compare_contracts_with_llm(reference_text, new_text, reference_pdf_abs, new_input_abs, args.verbose)
                    else:
                        print("Error: OpenAI package not installed. Please install with: pip install openai")
                        print("Then set your API key with: export OPENAI_API_KEY=your_api_key")
                else:
                    # Use traditional comparison
                    compare_texts(reference_text, new_text, reference_pdf_abs, new_input_abs, args.verbose)
        
        # Check if the input is a directory
        elif os.path.isdir(new_input):
            print(f"Comparing with contracts in directory: {os.path.basename(new_input_abs)}")
            for file_name in os.listdir(new_input):
                if file_name.endswith(".pdf"):
                    new_pdf_path = os.path.join(new_input, file_name)
                    new_pdf_path_abs = os.path.abspath(new_pdf_path)
                    print(f"Reading revised contract: {os.path.basename(new_pdf_path_abs)}")
                    new_text = extract_text_from_pdf(new_pdf_path_abs, args.verbose)
                    if new_text:
                        if args.llm:
                            # Use LLM-enhanced comparison
                            if HAS_OPENAI:
                                compare_contracts_with_llm(reference_text, new_text, reference_pdf_abs, new_pdf_path_abs, args.verbose)
                            else:
                                print("Error: OpenAI package not installed. Please install with: pip install openai")
                                print("Then set your API key with: export OPENAI_API_KEY=your_api_key")
                        else:
                            # Use traditional comparison
                            compare_texts(reference_text, new_text, reference_pdf_abs, new_pdf_path_abs, args.verbose)
        else:
            print(f"Input {new_input} is not a valid PDF file or directory.")
    else:
        # Use the default directory name if no comparison argument
        new_contracts_dir = "new_contracts"
        if os.path.isdir(new_contracts_dir):
            new_contracts_dir_abs = os.path.abspath(new_contracts_dir)
            print(f"Comparing with contracts in directory: {os.path.basename(new_contracts_dir_abs)}")
            for file_name in os.listdir(new_contracts_dir):
                if file_name.endswith(".pdf"):
                    new_pdf_path = os.path.join(new_contracts_dir, file_name)
                    new_pdf_path_abs = os.path.abspath(new_pdf_path)
                    print(f"Reading revised contract: {os.path.basename(new_pdf_path_abs)}")
                    new_text = extract_text_from_pdf(new_pdf_path_abs, args.verbose)
                    if new_text:
                        if args.llm:
                            # Use LLM-enhanced comparison
                            if HAS_OPENAI:
                                compare_contracts_with_llm(reference_text, new_text, reference_pdf_abs, new_pdf_path_abs, args.verbose)
                            else:
                                print("Error: OpenAI package not installed. Please install with: pip install openai")
                                print("Then set your API key with: export OPENAI_API_KEY=your_api_key")
                        else:
                            # Use traditional comparison
                            compare_texts(reference_text, new_text, reference_pdf_abs, new_pdf_path_abs, args.verbose)
        else:
            print(f"No comparison target specified and directory {new_contracts_dir} not found.")
            print("Please provide a valid PDF file or directory as the second argument.")

if __name__ == "__main__":
    print("Starting contract language comparison...")
    main()