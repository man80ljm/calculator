import sys
from PyQt6.QtWidgets import QApplication
from ui import GradeAnalysisApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GradeAnalysisApp()
    ex.show()
    sys.exit(app.exec())