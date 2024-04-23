import easydict

config = easydict.EasyDict()

##############################################################################################
# 모바일넷 백본 설정
config.BACKBONE = easydict.EasyDict()
config.BACKBONE.LIGHTWEIGHT = True
