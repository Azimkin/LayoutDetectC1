from model import LayoutDetectC1


def main():
    test_model = LayoutDetectC1.load("out/LayoutDetectC1.pth")

    run = True
    while run:
        inp = input("Enter phrase: ")

        clazz, values = test_model.classify_phrase(inp)
        print('Class:', clazz)
        print('Values:', values)


if __name__ == '__main__':
    main()