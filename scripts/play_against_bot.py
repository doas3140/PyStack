
import sys
import os

os.chdir('..')
sys.path.append( os.path.join(os.getcwd(),'src') )

def run_browser():
    try:
        import webbrowser
        webbrowser.open('src/GUI/client/game.html')
    except:
        print('WARNING: python cannot open browser')
        print('you will need to open "src/GUI/client/game.html" yourself')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # if args then load either browser or server
        if sys.argv[1] == '--browser':
            run_browser()
        elif sys.argv[1] == '--server':
            from GUI.server import run_server
            run_server()
        else:
            raise( Exception('Bad arguments: use one of "--browser" or "--server" or no arguments (then uses both)') )
    else: # else run both
        from GUI.server import run_server
        run_browser()
        run_server()
