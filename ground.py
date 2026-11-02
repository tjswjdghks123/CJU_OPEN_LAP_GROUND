"""GitHub edition: ten-input synthetic LSTM showcase and criteria processing."""
import sys

if __name__ == "__main__":
    if not sys.argv[1:]:
        from simple_showcase import main as simple_main
        simple_main()
        sys.exit(0)
    if sys.argv[1:2] == ["--manual"]:
        from manual_sensor_server import main as manual_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        manual_main()
        sys.exit(0)
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
