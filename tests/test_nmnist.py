from gin.nmnist import nmnist

# def test_download():
#     _ = download()


def test_nmnist():
    print("Getting NMNIST")
    x = nmnist()

    print("Getting frames")
    _ = x.frames()

    # print("Getting events")
    # _ = x.events()
