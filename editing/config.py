CHECKPOINT_PATH = "./pretrained/stylegan2-ffhq-config-f.pt"
FACE_PARSER_CKP = "pretrained/BiSetNet.pth"
CORRECTION_PATH = "pretrained/correction.pt"
STATISTICS_PATH = ""
CLASSIFIER_CKP = "pretrained/Attribute_CelebAMask-HQ_40_classifier.pth"
RECORD_PATH = ""

MAIN_LAYER = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
TO_RGB_LAYER = [1, 4, 7, 10, 13, 16, 19, 22, 25]
NUM_MAIN_LAYER_CHANNEL = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32]
NUM_TO_RGB_CHANNEL = [512, 512, 512, 512, 512, 256, 128, 64, 32]
CHANNEL_MAP = dict(zip(MAIN_LAYER + TO_RGB_LAYER, NUM_MAIN_LAYER_CHANNEL + NUM_TO_RGB_CHANNEL))

SEMANTIC_REGION = dict(
    [
        ("background", (0,)),  # 0
        ("brow", (1, 2)),  # 1
        ("eye", (3, 4)),  # 2
        ("glass", (5,)),  # 3
        ("ear", (6, 7, 8)),  # 4
        ("nose", (9,)),  # 5
        ("mouth", (10,)),  # 6
        ("lips", (11, 12)),  # 7
        ("neck", (13, 14)),  # 8
        ("cloth", (15,)),  # 9
        ("hair", (16,)),  # 10
        ("hat", (17,)),  # 11
        ("face_up", (18,)),  # 12
        ("face_middle", (19,)),  # 13
        ("face_down", (20,)),  # 14
    ]
)

CELEBA_ATTRS = [
    "5_o_Clock_Shadow",  # 0 短胡子
    "Arched_Eyebrows",  # 1 弯眉毛
    "Attractive",  # 2 有吸引力
    "Bags_Under_Eyes",  # 3 眼袋
    "Bald",  # 4 秃顶
    "Bangs",  # 5 刘海
    "Big_Lips",  # 6 厚嘴唇
    "Big_Nose",  # 7 大鼻子
    "Black_Hair",  # 8 黑色头发
    "Blond_Hair",  # 9 金色头发
    "Blurry",  # 10 模糊
    "Brown_Hair",  # 11 棕色头发
    "Bushy_Eyebrows",  # 12 浓眉毛
    "Chubby",  # 13 胖的
    "Double_Chin",  # 14 双下巴
    "Eyeglasses",  # 15 眼镜
    "Goatee",  # 16 山羊胡子
    "Gray_Hair",  # 17 灰白头发
    "Heavy_Makeup",  # 18 浓妆
    "High_Cheekbones",  # 19 高颧骨
    "Male",  # 20 男性
    "Mouth_Slightly_Open",  # 21 嘴巴微张
    "Mustache",  # 22 胡子，髭
    "Narrow_Eyes",  # 23 小眼睛
    "No_Beard",  # 24 没有胡子
    "Oval_Face",  # 25 鸭蛋脸
    "Pale_Skin",  # 26 皮肤苍白
    "Pointy_Nose",  # 27 尖鼻子
    "Receding_Hairline",  # 28 发际线后移
    "Rosy_Cheeks",  # 29 红润双颊
    "Sideburns",  # 30 连鬓胡子
    "Smiling",  # 31 微笑
    "Straight_Hair",  # 32 直发
    "Wavy_Hair",  # 33 卷发
    "Wearing_Earrings",  # 34 戴耳环
    "Wearing_Hat",  # 35 戴帽子
    "Wearing_Lipstick",  # 36 涂唇膏
    "Wearing_Necklace",  # 37 戴项链
    "Wearing_Necktie",  # 38 戴领带
    "Young",  # 39 年轻
]
