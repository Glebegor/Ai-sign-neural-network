import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

updateWights = True

class NeuralDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([
            [0.39638528, 0.83412021, 0.44976857, 0.79088885, 0.475541,   0.69080353,
             0.47998214, 0.59232438, 0.4466801,  0.55657786, 0.45489421, 0.63056815,
             0.46881047, 0.55409831, 0.45592028, 0.58979583, 0.43914995, 0.62078047,
             0.43143108, 0.60719091, 0.44649422, 0.50471318, 0.45778063, 0.44278485,
             0.4667573,  0.38413748, 0.40467018, 0.60532385, 0.41885713, 0.52796906,
             0.4206982,  0.59310496, 0.41476402, 0.65132105, 0.3768149,  0.61369687,
             0.39251488, 0.54430205, 0.40096626, 0.58949119, 0.39854705, 0.63949096],
            [0.27518579, 0.75295287, 0.29925954, 0.65861928, 0.34165305, 0.58656579,
             0.37952757, 0.53326893, 0.39033961, 0.48054415, 0.37677729, 0.6204524,
             0.42855737, 0.64129698, 0.41094792, 0.65468657, 0.38800564, 0.65266222,
             0.38325533, 0.68321115, 0.43299174, 0.69716048, 0.41414288, 0.70758528,
             0.39175084, 0.70604295, 0.38463792, 0.74590278, 0.43309808, 0.75352108,
             0.41445559, 0.75895715, 0.39245701, 0.75581622, 0.38197616, 0.80412191,
             0.42542404, 0.80259645, 0.40701881, 0.80390424, 0.38775283, 0.8012898],
            [0.40726554, 0.5193212,  0.41856205, 0.59783977, 0.45680287, 0.67565227,
             0.49748373, 0.71837598, 0.51700193, 0.76419073, 0.50949216, 0.62014127,
             0.53654456, 0.62303752, 0.49980408, 0.61055624, 0.48882267, 0.60489345,
             0.51127082, 0.56451303, 0.53811508, 0.56850022, 0.49139512, 0.56702685,
             0.49012214, 0.56720239, 0.50975543, 0.51412719, 0.53577709, 0.51482475,
             0.49340859, 0.52114147, 0.49034578, 0.52347064, 0.50626951, 0.46321994,
             0.52374148, 0.46581668, 0.49401349, 0.47779328, 0.487367,   0.48142454],
            [0.30597982, 0.84276772, 0.36181945, 0.8304553,  0.40927392, 0.77750462,
            0.44278139, 0.72669399, 0.4677929,  0.67784667, 0.39617142, 0.6423912,
            0.4279176,  0.58385485, 0.44774911, 0.60422713, 0.45886356, 0.6417374,
            0.37825263, 0.61240703, 0.40880594, 0.52979857, 0.43871439, 0.48212087,
            0.46531466, 0.43997329, 0.35513902, 0.60258007, 0.37512681, 0.50162333,
            0.39575344, 0.43770796, 0.41741729, 0.38658637, 0.32750154, 0.6109094,
            0.32448345, 0.52540398, 0.33200884, 0.46610457, 0.34583297, 0.41512722],
            [0.38332561, 0.82561624, 0.43676084, 0.79258835, 0.47837359, 0.71628994,
             0.51146269, 0.65960824, 0.5446775,  0.64060974, 0.44247764, 0.59242398,
             0.46104294, 0.50159514, 0.47118837, 0.4427135,  0.47805545, 0.39136326,
             0.41121095, 0.57863754, 0.41978371, 0.47677493, 0.42370564, 0.41102174,
             0.42465228, 0.3542304,  0.38107926, 0.58816439, 0.37802908, 0.488592,
             0.37667632, 0.4224571,  0.37512395, 0.36307546, 0.35154659, 0.61794806,
             0.32626551, 0.55099684, 0.31008825, 0.50587958, 0.29721069, 0.46059364],
        ], dtype=torch.float32)

        self.y = torch.tensor([
            [1, 0, 0, 0, 0],  # Fuck
            [0, 1, 0, 0, 0],  # like
            [0, 0, 1, 0, 0],  # dislike
            [0, 0, 0, 1, 0],  # Ok
            [0, 0, 0, 0, 1] # Hello
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(42, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def SoftMax(self, x):
        max_id = 0
        for i in range(len(x)):
            if max(x) == x[i]:
                max_id = i

        if max_id == 0:
            return "Fuck"
        if max_id == 1:
            return "Like"
        if max_id == 2:
            return  "Dislike"
        if max_id == 3:
            return  "Okay"
        if max_id == 4:
            return "Hello"

dataset = NeuralDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for test_images, test_labels in dataloader:
    sample_image = test_images[0]    # Reshape them according to your needs.
    sample_label = test_labels[0]
    print(sample_image)

model = NeuralNetwork()
try:
    model.load_state_dict(torch.load("./dataset.pth"))
except:
    torch.save(model.state_dict(), "./dataset.pth")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    model.train()
    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Ensure labels are torch.long
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Make predictions
ans = model.forward(torch.tensor([
    [0, 0.7, 0.2, 0.4, 0.4, 0.3, 0.2, 0, -0.3, -0.1, -0.5, 0.2, -0.7, 0.3, -0.3, 0.3, -0.5, 0.5, -0.7, 0.5, -0.2, -0.2,
     -0.4, -0.3, -0.7, -0.3, -0.2, -0.5, -0.5, -0.5, -0.6, -0.6, -0.2, -0.7, -0.5, -0.7, 0.6, -0.4, 0.2, -0.6, -0.4, 0],
], dtype=torch.float32))

print(list(ans))

# Stream model
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    return image, results


def draw_style_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


cap = cv2.VideoCapture(1)  # Assuming 0 for the primary webcam
with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_style_landmarks(image, results)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if results.pose_landmarks != None:
            pose = []
            for res in results.pose_landmarks.landmark:
                test = np.array([res.x, res.y])
                pose.append(test)

        lh = np.array([[res.x, res.y] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            42 * 1)
        rh = np.array([[res.x, res.y] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            42 * 1)

        right_res = model.forward(torch.tensor(rh, dtype=torch.float32))
        print(rh)
        right_res = model.SoftMax(right_res)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Distance: {}'.format(right_res), (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Holistic', image)

        # model.forward(torch.tensor(lh, dtype=torch.float32))

    cap.release()
    cv2.destroyAllWindows()
