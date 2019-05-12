''' 1 - code for writing labeled data to file '''

from torchvision import save_iamge
import os

''' a little bit of set up '''
data_path  = "path/to/your/data"
raw_data   = datasets.ImageFolder(data_path, transform=transform)
dataloader = usual_stuff

# this depends on the torch version you use -- I'm on 0.4.1 on Prince
# the call is a bit different from like 0.8+, I think it's just datasets.find_classes(data_root)
classes, classes_to_ids = raw_data._find_classes(data_root)

ids_to_class = []
for class_name in classes_to_ids:
    index = classes_to_ids[class_name]
    ids_to_class[index] = class_name

''' your code goes here '''

labels = model(images)

''' here's most of what you need to write the images correctly '''
gen_path = "generated_imgs"
for img, id_label in zip(images, labels):
    class_label = ids_to_class[id_label]

    if not os.path.exists(gen_path + '/' + class_label):
        os.mkdir(gen_path + '/' + class_label)

    some_id = 'idk_something_meaningful' # I used epochs and batches_completed
    save_image(img.data, "{0}/{1}/{1}_{2}.png".format(gen_path, class_label, some_id)


''' 2 - code for mixing up batches between multiple sources '''

# this is apparently much more efficient than two train loaders and 
# concatenating the batches from each -- haven't tested

label_data_path = "path/to/supervised/train"
gen_data_path   = "path/to/gen/imgs"

batch_size = 32
dataloader = data.DataLoader(
                datasets.ConcatDataset(
                     datasets.ImageFolder(label_data_path, transform=transform),
                     datasets.ImageFolder(gen_data_path,   transform=transform),
                ),
                batch_size=train_batch_size,
                shuffle=True,
             )

''' your code goes here '''

for i, (input, target) in enumerate(dataloader):
    # more stuff

