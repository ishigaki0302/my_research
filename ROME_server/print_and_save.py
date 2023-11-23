def print_and_save(text, filename):
    # 画面にテキストを表示
    print(text)
    
    # ファイルにテキストを追記
    with open(filename, "a") as file:
        file.write(text + "\n")