import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from model.model import CLIP
from zero_shot.class_names_and_templates import imagenet_classes, imagenet_templates
from utils.simple_tokenizer import SimpleTokenizer
import argparse
from tqdm import tqdm
from datasets import load_dataset
from utils import set_seed, mkdir, load_config_file

MODEL_CONFIG_PATH = 'model/model_config.yaml'

# Unchanged function
def getWordnetId2ClassName(WordnetId2ClassName_filepath):
    '''
    For ImageNet-like datasets. To convert WordNetId to class name.
    '''
    WordnetId2ClassName = {}
    with open(WordnetId2ClassName_filepath) as fp:
        for line in fp.readlines():
            wordNetId = line.split()[0]
            class_name = line.split()[-1]
            className = ' '.join(class_name.split('_'))
            WordnetId2ClassName[wordNetId] = className
    return WordnetId2ClassName

# Unchanged function
def tokenize(texts, tokenizer, context_length=77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

# Unchanged function
def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts, tokenizer).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

# Unchanged function
def evaluate(model, eval_dataloader, dataset_classes, WordnetId2ClassName, tokenizer, device, dataset_type):
    top_1_correct = 0
    top_5_correct = 0
    class_wise_top_1_correct = {}
    class_wise_top_5_correct = {}
    class_wise_total_examples = {}
    
    with torch.no_grad():
        if dataset_type == 'imagenet':
            templates = imagenet_templates
            print('DATASET',dataset_classes[0])
            # if class name is WordNetId
            # if dataset_classes[0][0] == 'n':
            #     classnames = [WordnetId2ClassName[c] for c in dataset_classes]
            # if class name is just index (for ImageNetV2)
            # else:
            classnames = [imagenet_classes[int(c)] for c in dataset_classes]
            
        elif dataset_type == 'cifar':
            templates = ["a photo of a {}."]
            classnames = [classname for classname in dataset_classes]
           
        zeroshot_weights = zeroshot_classifier(model, classnames, templates, tokenizer, device)
          
        for step, (images, labels) in enumerate(tqdm(eval_dataloader)):
            image_input = images.to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
            
            # top 5 predictions
            values, indices = similarity[0].topk(5)

            label_class = classnames[labels.item()]
            
            if label_class not in class_wise_total_examples:
                class_wise_top_1_correct[label_class] = 0
                class_wise_top_5_correct[label_class] = 0                
                class_wise_total_examples[label_class] = 0
            
            class_wise_total_examples[label_class] += 1

            if labels.item() == indices[0] :
                top_1_correct += 1
                class_wise_top_1_correct[label_class] += 1
                
            if labels.item() in indices:
                top_5_correct += 1
                class_wise_top_5_correct[label_class] += 1 

        top_1_accuracy = 100 * top_1_correct / len(eval_dataloader)
        top_5_accuracy = 100 * top_5_correct / len(eval_dataloader)

        class_wise_top_1_accuracy = {class_name: 100 * class_wise_top_1_correct[class_name] / class_wise_total_examples[class_name] for class_name in class_wise_top_1_correct}
        class_wise_top_5_accuracy = {class_name: 100 * class_wise_top_5_correct[class_name] / class_wise_total_examples[class_name] for class_name in class_wise_top_5_correct}
        
        return top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy

# Unchanged function
def save_accuracies(eval_result_output_file_path, top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy):
    with open(eval_result_output_file_path, "w") as fp:
        fp.write(f"Top 1 accuracy = {round(top_1_accuracy, 2)} %\n")
        fp.write(f"Top 5 accuracy = {round(top_5_accuracy, 2)} %\n")
        fp.write("----------------------------------------------------- \n")
        fp.write(f"Class wise accuracies (in %)\n\n")
        fp.write("{:<10}{:<10}{:<20}\n".format("Top-1", "Top-5", "Class name"))

        for class_name in class_wise_top_1_accuracy.keys():
            fp.write("{:<10}{:<10}{:<20}\n".format(round(class_wise_top_1_accuracy[class_name], 2), round(class_wise_top_5_accuracy[class_name], 2), class_name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="path of trained checkpoint")
    parser.add_argument("--dataset_type", default="imagenet", type=str, required=True, help="Type of eval dataset. 'imagenet' : for imagenet-like dataset / 'cifar' for CIFAR-like datasets")
    parser.add_argument("--WordnetId2ClassName_filepath", default="zero_shot/WordNetId2ClassName.txt", type=str, help="txt file containing WordNetId to class name mapping")
    args = parser.parse_args()

    zero_shot_eval_output_dir = "zero-shot-eval"
    mkdir(path=zero_shot_eval_output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_config = load_config_file(MODEL_CONFIG_PATH)

    # Image transform and text tokenizer
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),
    ])

    tokenizer = SimpleTokenizer()

    # Create RN50 CLIP model
    # model_params = dict(model_config.RN50)
    # model_params['vision_layers'] = tuple(model_params['vision_layers'])
    # model_params['vision_patch_size'] = None
    # model = CLIP(**model_params)
     # Creating ViT CLIP model
    model_params = dict(model_config.ViT)
    model_params['vision_patch_size'] = model_params['vision_patch_size']  # e.g., 16
    model_params['vision_layers'] = model_params['vision_layers']  # e.g., 12
    model = CLIP(**model_params)

    # Load trained weights
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if args.dataset_type == 'cifar':
        # Load CIFAR-10 from Hugging Face
        dataset = load_dataset("cifar10")
        # Load CIFAR-100 from Hugging Face
        # dataset = load_dataset('cifar100')

        print(dataset["test"][0]) 
        # Apply transformation and prepare DataLoader
        #cifar10
        imageDataset = [(transform(item["img"]), item["label"]) for item in dataset["test"]]
        #cifar100
        # imageDataset = [(transform(item["img"]), item["fine_label"]) for item in dataset["test"]]

        eval_dataloader = DataLoader(imageDataset, sampler=SequentialSampler(imageDataset), batch_size=1)

        # Get class names from CIFAR-10
        dataset_classes = dataset["test"].features["label"].names
        print("CIFAR-10 class names:", dataset_classes)

        # Get class names from CIFAR-100
        # dataset_classes = dataset["test"].features["fine_label"].names
        # print("CIFAR-100 class names:", dataset_classes)

    elif args.dataset_type == 'imagenet':
        # Load ImageNetV2 from Hugging Face
        dataset = load_dataset("clip-benchmark/wds_imagenetv2")

        print(dataset["test"][0]) 

        # Apply transformation and prepare DataLoader for ImageNetV2
        imageDataset = [(transform(item["webp"]), item["cls"]) for item in dataset["test"]]
        eval_dataloader = DataLoader(imageDataset, sampler=SequentialSampler(imageDataset), batch_size=1)

        dataset_classes = list(set(item["cls"] for item in dataset["test"]))  # Get all unique class labels
        # print("ImageNetV2 class names (numeric):", dataset_classes)

    else:
        print("Provide dataset type as either 'imagenet' or 'cifar'")
        return

    # Now evaluate
    top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy = evaluate(
        model, eval_dataloader, dataset_classes, {}, tokenizer, device, args.dataset_type
    )
    
    print("Top 1 accuracy:", round(top_1_accuracy, 2))
    print("Top 5 accuracy:", round(top_5_accuracy, 2))

    eval_result_output_file_path = os.path.join(zero_shot_eval_output_dir, f'{args.dataset_type}.txt')
    save_accuracies(eval_result_output_file_path, top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy)

    print("--------------------")
    print("Check this for class-wise accuracies: ", eval_result_output_file_path)

if __name__ == "__main__":
    main()
