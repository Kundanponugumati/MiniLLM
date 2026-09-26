"""Launch MiniLLM's terminal chat using the saved best checkpoint."""
import sys

from inference import main


if __name__ == "__main__":
    sys.argv.insert(1, "--chat")
    main()
