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

            [0.31713703, 0.80451977, 0.36821315, 0.72367269, 0.39025468, 0.62704748,
             0.4132821,  0.55824882, 0.43511748, 0.49033371, 0.34050134, 0.50182164,
             0.37032798, 0.41579938, 0.40496469, 0.42423931, 0.43270361, 0.45592818,
             0.3266111,  0.49276465, 0.34376237, 0.37408489, 0.37754411, 0.31742662,
             0.40958762, 0.28629136, 0.31057712, 0.51101679, 0.32194415, 0.38936734,
             0.34872371, 0.31694782, 0.3756105,  0.26281625, 0.2956225,  0.54541296,
             0.28948522, 0.44732532, 0.30185217, 0.37737787, 0.32168981, 0.31894895],

            [0.38332561, 0.82561624, 0.43676084, 0.79258835, 0.47837359, 0.71628994,
             0.51146269, 0.65960824, 0.5446775,  0.64060974, 0.44247764, 0.59242398,
             0.46104294, 0.50159514, 0.47118837, 0.4427135,  0.47805545, 0.39136326,
             0.41121095, 0.57863754, 0.41978371, 0.47677493, 0.42370564, 0.41102174,
             0.42465228, 0.3542304,  0.38107926, 0.58816439, 0.37802908, 0.488592,
             0.37667632, 0.4224571,  0.37512395, 0.36307546, 0.35154659, 0.61794806,
             0.32626551, 0.55099684, 0.31008825, 0.50587958, 0.29721069, 0.46059364],
            # Heart
            # [0.29861689, 0.83732975, 0.35240936, 0.83189315, 0.39840096, 0.79277629,
            #  0.4349837,  0.77650118, 0.46607849, 0.78679049, 0.39127088, 0.64870214,
            #  0.41149259, 0.58096129, 0.43322253, 0.57156789, 0.45101151, 0.582389,
            #  0.37786227, 0.63752371, 0.39952165, 0.54978365, 0.42857224, 0.54995513,
            #  0.4484528,  0.57897884, 0.36275324, 0.63581002, 0.38415697, 0.551651,
            #  0.41445661, 0.55462265, 0.43642959, 0.58279109, 0.34878185, 0.64333302,
            #  0.36809424, 0.57673818, 0.39330027, 0.56794709, 0.41620287, 0.58112198]

        ], dtype=torch.float32)

        self.y = torch.tensor([
            [1, 0, 0, 0, 0],  # Fuck
            [0, 1, 0, 0, 0],  # like
            [0, 0, 1, 0, 0],  # dislike
            [0, 0, 0, 1, 0],  # Ok
            [0, 0, 0, 0, 1] # Hello
            # [0, 0, 0, 0, 0, 1],  # Heart
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
        # if x[max_id]<0:
        #     return "Nothing"
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
        # if max_id == 5:
        #     return "Heart"


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
    num_epochs = 10000
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Ensure labels are torch.long
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    pass



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
        print(right_res)

        right_res = model.SoftMax(right_res)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'symbol: {}'.format(right_res), (10, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Holistic', image)

        # model.forward(torch.tensor(lh, dtype=torch.float32))

    cap.release()
    cv2.destroyAllWindows()
