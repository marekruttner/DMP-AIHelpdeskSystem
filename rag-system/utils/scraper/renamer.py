import os
import unidecode


def rename_files_in_directory(directory):
    """
    Rename all markdown files in the directory, converting non-ASCII characters to their nearest US English equivalents.
    :param directory: Directory containing markdown files.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            original_path = os.path.join(directory, filename)

            # Remove special characters and convert non-ASCII characters to ASCII equivalents
            base_name = os.path.splitext(filename)[0]
            ascii_name = unidecode.unidecode(base_name)

            # Replace spaces with underscores and remove special characters
            ascii_name = ascii_name.replace(" ", "_")
            ascii_name = ''.join(char for char in ascii_name if char.isalnum() or char == "_")

            # Form the new filename with .md extension
            new_filename = ascii_name + ".md"
            new_path = os.path.join(directory, new_filename)

            # Rename the file if the name has changed
            if original_path != new_path:
                print(f"Renaming '{filename}' to '{new_filename}'")
                os.rename(original_path, new_path)


if __name__ == "__main__":
    # Directory containing the markdown files
    directory = "scraped_markdown"  # Adjust this path as needed
    rename_files_in_directory(directory)
