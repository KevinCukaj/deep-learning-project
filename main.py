import demucs
import demucs.separate


def main():
    print("Hello from deep-learning-project!")
    
    # separate source in bass, drums, other, vocals
    # the pre-trained model is mdx_extra
    demucs.separate.main(["--mp3", "-n", "mdx_extra", "./test-tracks/track1.mp3"])



if __name__ == "__main__":
    main()
