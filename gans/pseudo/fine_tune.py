
'''
train_data = datasets.ImageFolder(train_dir, transform=transform)
classes, classes_to_idx = raw_data._find_classes(data_root)
idx_to_class = {}
for class_name in classes_to_idx:
    index = classes_to_idx[class_name]
    idx_to_class[index] = class_name
print("num classes: ", len(classes))

train_loader = torch.utils.data.DataLoader(
    raw_data,
    batch_size=batch_size,
    shuffle=True,
)
print("loaded train data, num batches=", len(train_loader), ", batch_size=", batch_size)

val_data = datasets.ImageFolder(train_dir, transform=transform)
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True,
)
'''
