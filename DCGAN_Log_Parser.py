import json
import matplotlib.pyplot as plt

# Specify the file path
file_path = 'data.txt'  # Replace with the actual file path

# filtered_list = []
# # Open the file for reading
# with open(file_path, 'r') as file:
#     # Read the entire content of the file into a variable
#     file_content = file.read()
#
# # Now you can work with the content of the file as needed
# # For example, you can print it to the console
# # print(file_content)
#
# # You can also split the content into lines if it's a multi-line file
# lines = file_content.split('\n')
# for line in lines:
#     # Process each line as needed
#     if "Sva" not in line:
#         if "[10/25]" not in line:
#             filtered_list.append(line)
#
#
#
# with open(file_path, 'w') as file:
#     file.writelines(filtered_list)

# Initialize separate lists for each type of element
epochs = []
steps = []
disc_losses = []
gen_losses = []
D_x_values = []
D_G_z_values = []

with open(file_path, 'r') as file:
    file_content = file.read()
    lines = file_content.split('\n')
    num = 1
    for line in lines:
        # Split the line based on commas and spaces
        elements = line.split(', ')



        # Iterate through the elements and parse them
        for element in elements:
            if element.startswith("Epoch"):
                epochs.append(num)
            elif element.startswith("Step"):
                step_info = element.split(' ')[1]  # Get the "[20/25]" part
                steps.append(step_info)
            elif element.startswith("disc_loss"):
                disc_loss = float(element.split(': ')[1])  # Get the disc_loss value
                disc_losses.append(disc_loss)
            elif element.startswith("gen_loss"):
                gen_loss = float(element.split(': ')[1])  # Get the gen_loss value
                gen_losses.append(gen_loss)
            elif element.startswith("D(x)"):
                D_x_value = float(element.split(': ')[1])  # Get the D(x) value
                D_x_values.append(D_x_value)
            elif element.startswith("D (G(z))"):
                D_G_z_value = float(element.split(': ')[1])  # Get the D (G(z)) value
                D_G_z_values.append(D_G_z_value)

        num +=1


# Print or use the separate lists as needed
print("Epochs:", epochs)
print("Steps:", steps)
print("Disc Losses:", disc_losses)
print("Gen Losses:", gen_losses)
print("D(x) Values:", D_x_values)
print("D(G(z)) Values:", D_G_z_values)



# Plot disc_losses vs gen_losses
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(disc_losses, label='Discriminator')
plt.plot(gen_losses, label='Generator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses')
plt.legend()

# Plot D(x) vs D(G(z))
plt.subplot(1, 3, 2)
plt.plot(epochs[0:100],D_x_values[0:100], label='Real Scores')
plt.plot(epochs[0:100],D_G_z_values[0:100], label='Fake Scores')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Scores')
plt.legend()

# Plot D(x) vs D(G(z))
plt.subplot(1, 3, 3)
plt.plot(epochs[100:],D_x_values[100:], label='Real Scores')
plt.plot(epochs[100:], D_G_z_values[100:], label='Fake Scores')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Scores')
plt.legend()
# Adjust layout and display the plots
# plt.tight_layout()
plt.show()
