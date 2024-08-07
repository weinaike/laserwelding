from . import s3d, i3d, s3d_resnet, i3d_resnet, resnet, inception_v1, torch_s3d, resnet_multihead

from torchvision.models.video.s3d import S3D_Weights

MODEL_TABLE = {
    's3d': s3d,
    'i3d': i3d,
    's3d_resnet': s3d_resnet,
    'i3d_resnet': i3d_resnet,
    'resnet': resnet,
    'inception_v1': inception_v1,
    'torch_s3d': torch_s3d,
    'resnet_multihead': resnet_multihead
}


def build_model(args, test_mode=False):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    if args.backbone_net ==  "torch_s3d":
        model = torch_s3d(num_classes=args.num_classes, dropout=args.dropout, weights=S3D_Weights.KINETICS400_V1)
        #model = torch_s3d(num_classes=args.num_classes, dropout=args.dropout)
    else:
        model = MODEL_TABLE[args.backbone_net](**vars(args))

    network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net
    arch_name = "{dataset}-{modality}-{arch_name}".format(
        dataset=args.dataset, modality=args.modality, arch_name=network_name)
    arch_name += "-f{}".format(args.groups)

    # add setting info only in training
    if not test_mode:
        arch_name += "-{}{}-bs{}-e{}".format(args.lr_scheduler, "-syncbn" if args.sync_bn else "",
                                             args.batch_size, args.epochs)
    return model, arch_name
