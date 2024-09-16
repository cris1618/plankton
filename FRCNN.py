import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import Adam
from torch import nn
from torcheval.metrics import MulticlassAUPRC
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
import math
import sys
from sklearn.metrics import precision_recall_curve, average_precision_score


# Dataset class
class PlanktonDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

        # Group the annotations by image
        self.image_annotations = self.annotations.groupby('file')

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        img_name = list(self.image_annotations.groups.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get all annotations for this image
        annotations = self.image_annotations.get_group(img_name)

        boxes = []
        labels = []
        for _, row in annotations.iterrows():
            box_str = row['box'].replace(' ', ',').replace(',,', ',').strip('[]')
            box = [int(b) for b in box_str.split(',') if b.isdigit()]
            boxes.append(box)
            labels.append(int(row['label']) + 1)

        if self.transform:
            image = self.transform(image)

        # Convert to torch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return image, target


# Manually define FastRCNNPredictor
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def collate_fn(batch):
    return tuple(zip(*batch))


def visualize_images_with_boxes(images, targets, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        ax = axes[i]
        ax.imshow(image)

        for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
            box = box.cpu().numpy()
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, str(label.item()), color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
    plt.show()


def visualize_predictions(model, dataloader, device, num_images=5, confidence_threshold=0.5):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    transform = transforms.ToPILImage()

    for idx, (images, targets) in enumerate(dataloader):
        if idx >= num_images:
            break

        image = images[0].to(device)
        with torch.no_grad():
            prediction = model([image])

        image = image.cpu()
        image = transform(image)

        ax = axes[idx]
        ax.imshow(image)

        for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
            if score >= confidence_threshold:
                box = box.cpu().numpy()
                x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y, str(label.item()), color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
    plt.show()


def evaluate(model, dataloader, device):
    model.eval()
    detection_results = []
    ground_truths = []

    for images, targets in tqdm(dataloader):
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            outputs = model(images)

        # Collect ground truths and predictions
        for target, output in zip(targets, outputs):
            target_boxes = torch.tensor(target['boxes'].cpu().numpy())
            target_labels = torch.tensor(target['labels'].cpu().numpy())
            pred_boxes = torch.tensor(output['boxes'].cpu().numpy())
            pred_scores = torch.tensor(output['scores'].cpu().numpy())
            pred_labels = torch.tensor(output['labels'].cpu().numpy())

            ground_truths.append({'boxes': target_boxes, 'labels': target_labels})
            detection_results.append({'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels})

    return ground_truths, detection_results


def compute_map(detection_results, ground_truths):
    mean_ap_metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    mean_ap_metric.update(detection_results, ground_truths)
    mAP = mean_ap_metric.compute()

    return mAP['map'].item()


def compute_map_class(detection_results, ground_truths):
    mean_ap_metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    mean_ap_metric.update(detection_results, ground_truths)
    mAP = mean_ap_metric.compute()
    # Extract and handle class-specific metrics
    map_per_class = mAP['map_per_class']
    classes = mAP['classes']
    if map_per_class.nelement() > 1 and map_per_class[0].item() != -1:
        for i, ap in enumerate(map_per_class):
            if ap.item() == -1:
                print(f"Class {classes[i].item()} has no positive samples.")
            else:
                print(f"Average Precision for class {classes[i].item()-1}: {ap.item():.4f}")
    else:
        print("Class metrics are disabled.")

    return map_per_class


"""def compute_metrics(ground_truths, detection_results, num_classes):
    # Compute mean Average Precision (mAP)
    mean_ap_metric = MeanAveragePrecision(iou_thresholds=[0.5])
    mean_ap_metric.update(detection_results, ground_truths)
    mean_ap = mean_ap_metric.compute()

    # Compute Multiclass AUPRC
    multiclass_auprc = MulticlassAUPRC(num_classes=num_classes)

    for gt, det in zip(ground_truths, detection_results):
        gt_labels = gt['labels']
        pred_labels = det['labels']

         # Debug print statements to inspect shapes and values
        print(f"gt_labels shape: {gt_labels.shape}, values: {gt_labels}")
        print(f"pred_labels shape: {pred_labels.shape}, values: {pred_labels}")

        try:
            multiclass_auprc.update(pred_labels, gt_labels)
        except Exception as e:
            print(f"Error updating AUPRC: {e}")

    auprc = multiclass_auprc.compute()

    return mean_ap #, auprc"""

if __name__ == '__main__':
    # Model Checkpoint
    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    dir = os.getcwdb().decode('utf-8')
    folder_path = os.path.join(dir, 'CROPPED')
    csv_file_path = os.path.join(dir, 'solace_crop.csv')

    # Load the annotations
    annotations = pd.read_csv(csv_file_path)
    train_img_from_id = [46, 33, 49, 7, 3, 2, 1, 58, 44, 6, 54, 59, 39, 4, 10, 32, 16, 14, 11, 19, 64, 36, 17, 63, 23,
                         66, 43, 5, 65, 42, 26, 0, 69, 21, 27, 56, 12, 50, 37, 55, 61, 52, 48, 24, 8, 62, 47, 45, 53,
                         20, 67, 30, 40, 25, 31, 15, 18, 57, 41, 34, 68]
    test_img_from_id = [70, 51, 22, 9, 71, 60, 35]

    # Filter the DataFrame
    train_annotations = annotations[annotations['img_from_id'].isin(train_img_from_id)]
    test_annotations = annotations[annotations['img_from_id'].isin(test_img_from_id)]

    # Sample a fraction of the dataset (0.3%)
    train_sample = train_annotations.sample(frac=1, random_state=42)
    test_sample = test_annotations.sample(frac=1, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Try to normalize images
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #It greatly affects image's pixel, DON'T USE IT!
    ])

    # Create the datasets
    train_dataset = PlanktonDataset(annotations=train_sample, img_dir=folder_path, transform=transform)
    test_dataset = PlanktonDataset(annotations=test_sample, img_dir=folder_path, transform=transform)

    """#View a sample of an image
    sample = train_dataset[37]
    img_int = torch.tensor(sample[0] * 255).byte().clone().detach()
    plt.imshow(draw_bounding_boxes(img_int, sample[1]['boxes'], [str(label.item()) for label in sample[1]['labels']], width=4).permute(1,2,0))
    plt.show()"""

    # Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        # Decreased for testing purpose on my laptop (It was 1, but now i put it back at 5 in order to run it on the cluster)
        shuffle=True,
        num_workers=8,  # Decreased for testing purpose on my laptop
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,  # Decreased for testing purpose on my laptop
        shuffle=False,
        num_workers=8,  # Decreased for testing purpose on my laptop
        collate_fn=collate_fn
    )

    # VISUALIZE IMAGES INPUTTED IN THE MODEL BEFORE TRAINING
    images, targets = next(iter(train_loader))
    # visualize_images_with_boxes(images, targets)

    # Load the pre-trained Faster R-CNN model
    # model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
    path = os.getcwdb().decode('utf-8') + '/frcnn_sgdlr0001_batch4_noaugm.pkl'
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    #model.load_state_dict(torch.load(path))
    num_classes = 19
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to GPU if possible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    """# Load model weights if available
    checkpoint_path = 'model_checkpoints/fasterrcnn_checkpoint_epoch_1.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")"""

    # Define lists to store loss and mAP values
    train_losses = []
    val_losses = []
    mAP_train_values = []
    mAP_val_values = []

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)


    # Define functions to run 1 epoch
    # Training function
    def train_one_epoch(model, optimizer, loader, device, epoch):
        model.to(device)
        model.train()
        running_loss = 0.0

        all_losses = []
        all_losses_dict = []

        for images, targets in tqdm(loader):
            images = list(image.to(device) for image in images)
            # Changed 'v.to(device)' with torch.tensor(v).to(device)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
            loss_value = losses.item()
            running_loss += loss_value

            all_losses.append(loss_value)
            all_losses_dict.append(loss_dict_append)

            """if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)"""

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
        print(
            "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
                all_losses_dict['loss_classifier'].mean(),
                all_losses_dict['loss_box_reg'].mean(),
                all_losses_dict['loss_rpn_box_reg'].mean(),
                all_losses_dict['loss_objectness'].mean()
            ))

        epoch_loss = running_loss / len(loader)
        # epoch_loss = np.mean(all_losses_dict[all_losses])
        return epoch_loss

    def test_one_epoch(model, loader, device, epoch):
        model.to(device)
        model.train()
        running_loss = 0.0

        all_losses = []
        all_losses_dict = []

        with torch.no_grad():
            for images, targets in tqdm(loader):
                images = list(image.to(device) for image in images)
                # Changed 'v.to(device)' with torch.tensor(v).to(device)
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                loss_value = losses.item()
                running_loss += loss_value

                all_losses.append(loss_value)
                all_losses_dict.append(loss_dict_append)

                """if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict)
                    sys.exit(1)"""

        all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
        print(
            "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
                all_losses_dict['loss_classifier'].mean(),
                all_losses_dict['loss_box_reg'].mean(),
                all_losses_dict['loss_rpn_box_reg'].mean(),
                all_losses_dict['loss_objectness'].mean()
            ))

        epoch_loss = running_loss / len(loader)
        # epoch_loss = np.mean(all_losses_dict[all_losses])
        return epoch_loss

    def save_epochs(file, epochs, data_train, data_val):
        with open(file, "w") as f:
            f.write('\n\n')
            for i in range(epochs):
                f.write(f'{i} \t {data_train[i]} \t {data_val[i]}\n')

    def save_map_per_class(file, map_per_class):
        with open(file, "w") as f:
            f.write('\n\n')
            if map_per_class.nelement() > 1 and map_per_class[0].item() != -1:
                for i, ap in enumerate(map_per_class):
                    if ap.item() == -1:
                        f.write(f"Class {i} has no positive samples.")
                    else:
                        f.write(f"Average Precision for class {i}: {ap.item():.4f}")

    # TRAINING LOOP
    num_epochs = 4
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_losses.append(train_loss)

        ground_truths_train, detection_results_train = evaluate(model, train_loader, device)
        ground_truths_test, detection_results_test = evaluate(model, test_loader, device)
        val_loss = test_one_epoch(model, test_loader, device, epoch)
        val_losses.append(val_loss)
        mAP_train = compute_map(detection_results_train, ground_truths_train)
        mAP_train_values.append(mAP_train)
        mAP_test = compute_map(detection_results_test, ground_truths_test)
        mAP_val_values.append(mAP_test)
        print("mAP_train = {:.6f} \t maP_test = {:.6f} \n".format(mAP_train, mAP_test))

    # evaluate and compute AP for each class for training set
    ground_truths_train, detection_results_train = evaluate(model, train_loader, device)
    map_per_class_train = compute_map_class(detection_results_train, ground_truths_train)
    # Print mAP per class if available
    print(map_per_class_train)

    # evaluate and compute AP for each class for the testing set
    ground_truths_eval, detection_results_eval = evaluate(model, test_loader, device)
    map_per_class_eval = compute_map_class(detection_results_eval, ground_truths_eval)
    # Print mAP per class if available
    print(map_per_class_eval)

    torch.save(model.state_dict(), path)

    save_epochs('losses_sgdlr0001_batch4_noaugm', num_epochs, train_losses, val_losses)
    save_epochs('mAP_sgdlr0001_batch4_noaugm', num_epochs, mAP_train_values, mAP_val_values)
    save_map_per_class('mAP_class_train_sgdlr0001_batch4_noaugm', map_per_class_train)
    save_map_per_class('mAP_class_test_sgdlr0001_batch4_noaugm', map_per_class_eval)


    """if 'map_per_class' in map_metrics and map_metrics['map_per_class'].dim() > 0:
        print("mAP per class:")
        for idx, map_score in enumerate(map_metrics['map_per_class']):
            print(f"Class {idx}: {map_score.item()}")
    else:
        print("mAP per class could not be computed.")"""

    """# Plotting
    epochs = np.arange(1, num_epochs + 1)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10, 8))
    plt.suptitle("Loss vs Accuracy")

    # Plot training and validation loss
    ax[0].plot(epochs, train_losses, label='Training Loss')
    ax[0].plot(epochs, val_losses, label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].legend()

    # Plot mAP
    ax[1].plot(epochs, mAP_train_values, label='Training mAP')
    ax[1].plot(epochs, mAP_val_values, label='Validation mAP')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('mAP')
    ax[1].set_title('Mean Average Precision (mAP)')
    ax[1].legend()
    plt.show()"""

    """# Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'fasterrcnn_checkpoint_epoch_{epoch + 1}.pth'))
        print(f"Saved checkpoint for epoch {epoch + 1}")"""

    # VISUALIZE PREDICTIONS OF THE MODEL
    # visualize_predictions(model, test_loader, device, confidence_threshold=0.5)

    """#Evaluation
    ground_truths, detection_results = evaluate(model, test_loader, device)
    num_classes = 19  

    mAP, auprc = compute_metrics(ground_truths, detection_results, num_classes)

    #print(f"Mean Average Precision (mAP): {mAP['map'].item():.4f}")
    print(f"Mean Average Precision (mAP): {mAP}")
   # If map_per_class is available, print per-class AP
    if 'map_per_class' in mAP and mAP['map_per_class'].numel() > 1:
        for i, ap in enumerate(mAP['map_per_class']):
            print(f"Average Precision for class {i + 1}: {ap:.4f}")
    else:
        print("Per-class mAP is not available or not valid.")

    #print(f"Multiclass AUPRC: {auprc}")


    #Evaluation
    #metric_ap = MulticlassAUPRC(num_classes=num_classes, average=None, device=device)
    metric_map = MeanAveragePrecision()

    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            # Same thing that i did in the training loop
            images = list(image.to(device) for image in images)
            #Changed 'v.to(device)' with torch.tensor(v).to(device)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            # Concatenate scores and labels
            preds_scores = [output['scores'].to(device) for output in outputs]
            true_labels = [target['labels'].to(device) for target in targets]

            #if len(preds_scores) > 0 and len(true_labels) > 0:
            #   metric_ap.update(preds_scores, true_labels)
            metric_map.update(preds_scores, true_labels)
        #ap_per_class = metric_ap.compute()
        #print(f"Average Precision (AP) per class: {ap_per_class}")

        #MEAN AVERAGE PRECISION PER CLASS
        mAP = metric_map.compute()
        print(f"The mean Average Precision (mAP) of the model is {mAP}")"""

    """# Training loop with progress tracking
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            # Use of List Comprehensions to create list of images and targets so that you don't need to use
            # the .squeeze(0) function the batch dimension is mantained and so now the model can handle batch sizes greater than 1
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets)
            loss_dict = outputs
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{len(train_loader)}, Loss: {losses.item()}")

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'fasterrcnn_checkpoint_epoch_{epoch + 1}.pth'))
        print(f"Saved checkpoint for epoch {epoch + 1}")

    print("Training Completed!")

    # VISUALIZE PREDICTIONS OF THE MODEL
    visualize_predictions(model, test_loader, device, confidence_threshold=0.5)"""