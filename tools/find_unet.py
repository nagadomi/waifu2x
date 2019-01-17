def find_upcunet_v2():
    avg_pool=4
    print_mod = False
    check_mod = True
    print("cascade")
    
    for i in range(76, 512):
        print("-- {}".format(i))
        print_buf = []
        s = i
        # unet 1

        s = s - 4 # conv3x3x2
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_mod: print(s, s % 2, s % 4, s % 6, s % 8)
        if check_mod and s % avg_pool != 0:
            continue

        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        
        if print_mod: print(s, s % 2, s % 4, s % 6, s % 8)
        if check_mod and s % avg_pool != 0:
           continue
        s = s * 2 # up2x2
        s = s - 4 # conv3x3x2
        if print_mod: print(s, s % 2, s % 4, s % 6, s % 8)
        if check_mod and s % avg_pool != 0:
            continue
        s = s * 2 # up2x2

        # deconv
        s = s
        s = s * 2 - 4

        # unet 2
        s = s - 4 # conv3x3x2
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_mod: print(s, s % 2, s % 4, s % 6, s % 8)
        if check_mod and s % avg_pool != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_mod: print(s, s % 2, s % 4, s % 6, s % 8)
        if check_mod and s % avg_pool != 0:
            continue
        s = s * 2 # up2x2
        s = s - 4 # conv3x3x2
        if print_mod: print(s, s % 2, s % 4, s % 6, s % 8)
        if check_mod and s % avg_pool != 0:
            continue
        s = s * 2 # up2x2
        s = s - 2 # conv3x3 last
        #if s % avg_pool != 0:
        #    continue
        print("ok", i, s)

def find_upcunet():
    check_mod = True
    print_size = False
    print("cascade")
    
    for i in range(72, 512):
        print_buf = []
        s = i
        # unet 1

        s = s - 4 # conv3x3x2
        if print_size: print("1/2", s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_size: print("1/2",s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        
        s = s * 2 # up2x2
        if print_size: print("2x",s)
        s = s - 4 # conv3x3x2
        s = s * 2 # up2x2
        if print_size: print("2x",s)

        # deconv
        s = s - 2
        s = s * 2 - 4

        # unet 2
        s = s - 4 # conv3x3x2
        if print_size: print("1/2",s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_size: print("1/2",s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        s = s * 2 # up2x2
        if print_size: print("2x",s)
        s = s - 4 # conv3x3x2
        s = s * 2 # up2x2
        if print_size: print("2x",s)
        s = s - 2 # conv3x3
        s = s - 2 # conv3x3 last
        #if s % avg_pool != 0:
        #    continue
        print("ok", i, s, s/ i)
        
def find_cunet():
    check_mod = True
    print_size = False
    print("cascade")
    
    for i in range(72, 512):
        print_buf = []
        s = i
        # unet 1

        s = s - 4 # conv3x3x2
        if print_size: print("1/2", s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_size: print("1/2",s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        
        s = s * 2 # up2x2
        if print_size: print("2x",s)
        s = s - 4 # conv3x3x2
        s = s * 2 # up2x2
        if print_size: print("2x",s)

        s = s - 4
        #s = s * 2 - 4

        # unet 2
        s = s - 4 # conv3x3x2
        if print_size: print("1/2",s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        if print_size: print("1/2",s)
        if check_mod and s % 2 != 0:
            continue
        s = s / 2 # down2x2
        s = s - 4 # conv3x3x2
        s = s * 2 # up2x2
        if print_size: print("2x",s)
        s = s - 4 # conv3x3x2
        s = s * 2 # up2x2
        if print_size: print("2x",s)
        s = s - 2 # conv3x3
        s = s - 2 # conv3x3 last
        #if s % avg_pool != 0:
        #    continue
        print("ok", i, s, s / i)
        
#find_upcunet()
find_cunet()
