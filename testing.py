import serial
import time
import numpy as np

# Define the END_MARKER byte (0xFF)
END_MARKER = 0xFF


# Function to send data to STM32 device
def send_data(ser, values):
    # Convert the list of integers into a comma-separated string with each value as a 4-digit string
    data_string = ','.join(f"{int(value):04d}" for value in values)

    # Send the data string over UART
    ser.write(data_string.encode())  # Send as bytes

    # Send the end marker (0xFF) to indicate end of transmission
    ser.write(bytes([END_MARKER]))


# Function to read and store data from the serial port
def serial_monitor(ser, baudrate=115200, timeout=1, output_file="output_data.txt"):
    try:
        while True:  # Monitor for a max of 'max_duration' seconds
            with open(output_file, 'a') as file:  # Open file in append mode
                if ser.in_waiting > 0:
                    data = ser.readline()  # Read line (can change to .read(size) for byte-by-byte)
                    if data:
                        decoded_data = data.decode('utf-8').strip()
                        print(f"Received: {decoded_data}")
                        match decoded_data:
                            case "O[0] (1,1,1,1)1/float32 @0x20000234/4":
                                break

                            case "0":
                                break


                        # Write the received data to the output file
                        file.write(decoded_data + "\n")  # Append data to the file
                        file.flush()  # Ensure data is written immediately

        print("Finished monitoring for input.")

    except PermissionError:
        print("Error: Permission denied to access port. Please check if another program is using it.")
    except KeyboardInterrupt:
        print("\nExiting serial monitor.")
    except Exception as e:
        print(f"Error: {e}")


# Main function
def main():
    # Open the serial port (adjust with your actual port)
    port = 'COM12'  # Replace with your actual port
    baudrate = 115200  # Baud rate for the serial communication
    timeout = 1  # Timeout for reading serial data

    # Load the npz file containing x_test and y_test
    data = np.load(r'C:\Users\Admin\Downloads\data_PatientVectors_len_40_overlap_39-20241115T163229Z-001\data_PatientVectors_len_40_overlap_39\training_and_validation.npz')
    x_test = data['x_test']  # Shape: (281911, 40, 1)
    y_test = data['y_test']  # Shape: (281911,)

    try:
        print("Serial monitor started. Sending data...\n")

        # Open the serial port for data sending and receiving
        ser = serial.Serial(port, baudrate, timeout=timeout)

        # Loop through x_test and y_test and send data row by row
        for i in range(len(x_test)):

            # Start capturing the serial output after all data is sent (for 2 seconds)
            serial_monitor(ser)  # Start the serial monitor for a fixed duration
            # Extract the row from x_test and corresponding label from y_test
            x_values = x_test[i].flatten()  # Flatten to a 1D array (shape: (40,))
            y_value = y_test[i]  # Single value (shape: ())

            # Convert values to standard Python int to avoid np.int64 issues
            x_values = [int(value) for value in x_values]
            y_value = int(y_value)  # Ensure y_value is an integer

            # Combine the x_values (40 values) and y_value (1 value) into a single list
            combined_values = x_values + [y_value]  # 40 + 1 = 41 values

            # Send the combined values over serial
            send_data(ser, combined_values)

            # Optionally, print what was sent for debugging purposes`
            print(f"Sent data: {combined_values}")



    except KeyboardInterrupt:
        print("\nExiting serial monitor.")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Closed serial port.")


if __name__ == '__main__':
    main()
