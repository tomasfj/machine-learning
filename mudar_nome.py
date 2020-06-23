
import os 

def main(): 

    for count, filename in enumerate(os.listdir("./dataset/flor5/")): 
        dst ="flor5_" + str(count) + ".jpg"
        src ='./dataset/flor5/'+ filename 
        dst ='./dataset/flor5/'+ dst 

        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 

# Driver Code 
if __name__ == '__main__': 

    # Calling main() function 
    main()
