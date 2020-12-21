from webapp import app

def main():
    """
    Main flow.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()