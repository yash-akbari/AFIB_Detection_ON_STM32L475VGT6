import sys
import serial
import time
import numpy as np
import threading

# Global event to signal that the MCU is ready to receive data.
ready_event = threading.Event()

def uart_monitor(ser, log_file):
    """
    Continuously monitor the UART for lines from the MCU.
    For each received line:
      - If "Start" is detected, set the ready event.
      - If "Stop" is detected, clear the ready event.
    All received lines are printed to the console and logged to the file.
    """
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if line:
                print("MCU:", line)
                log_file.write(line + "\n")
                log_file.flush()

                # When the MCU sends "Start", allow sending a row.
                if "Start" in line:
                    ready_event.set()
                # When the MCU sends "Stop", wait for the next cycle.
                elif "Stop" in line:
                    ready_event.clear()
        except Exception as e:
            print("Error in uart_monitor:", e)
            break

def send_data(ser, values):
    """
    Send a list of 40 integer values as a comma-separated string.
    Each integer is formatted as a 4-digit number with leading zeros.
    A newline is appended to signal end-of-packet.
    """
    data_string = ','.join(f"{int(value):04d}" for value in values)
    ser.write(data_string.encode() + b'\n')
    # Brief pause to allow MCU processing
    time.sleep(0.001)

def main():
    port = 'COM11'            # Replace with your actual COM port.
    baudrate = 115200         # Must match the MCU's UART settings.
    timeout = 1               # Timeout for reading serial data.

    # Open the serial port.
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
    except Exception as e:
        print("Failed to open port:", e)
        return

    # Allow time for the port and MCU to initialize.
    time.sleep(2)

    # Open a log file for writing MCU output.
    try:
        log_file = open("output_data.txt", "a")
    except Exception as e:
        print("Failed to open log file:", e)
        ser.close()
        return

    # Start the monitoring thread to listen for MCU handshake messages.
    monitor_thread = threading.Thread(target=uart_monitor, args=(ser, log_file), daemon=True)
    monitor_thread.start()

    # Load your NPZ data. The file should contain an 'x_test' array with shape (N, 40, ...) or (N, 40).
    npz_path = r'training_and_validation.npz'
    try:
        data = np.load(npz_path)
    except Exception as e:
        print("Error loading data:", e)
        ser.close()
        log_file.close()
        return

    x_test = data['x_test']
    # Remove singleton dimensions if needed. We expect each row to be of length 40.
    x_test = np.squeeze(x_test)  # Expected shape: (N, 40)
    total_rows = x_test.shape[0]
    print(f"Total rows available: {total_rows}")

    row_index = 0

    # Main loop: for each row, wait for a "Start" signal, send one row, then wait for "Stop" signal.
    while row_index < total_rows:
        # Wait indefinitely until the MCU signals it's ready by sending "Start".
        print("Waiting for 'Start' signal from MCU...")
        ready_event.wait()
        # Once ready_event is set, send one row.
        row = x_test[row_index].flatten()
        row = [int(value) for value in row]
        print(f"Sending row {row_index}: {row}")
        send_data(ser, row)
        row_index += 1

        # After sending, wait for the MCU to send "Stop" to clear the ready event before sending the next row.
        while ready_event.is_set():
            time.sleep(0.001)

    print("Finished sending all rows.")
    ser.close()
    log_file.close()

if __name__ == '__main__':
    main()