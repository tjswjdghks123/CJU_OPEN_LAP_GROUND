"""GitHub edition: ten-input synthetic LSTM showcase and criteria processing."""
import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["--criteria"]:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from device_server import main
    elif sys.argv[1:2] == ["stdin"]:
        from device_server import main
    else:
        if sys.argv[1:2] == ["--showcase"]:
            sys.argv = [sys.argv[0], *sys.argv[2:]]
        from demo_showcase import main
    main()
