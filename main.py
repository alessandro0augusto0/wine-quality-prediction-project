# ==============================================================================
# TRABALHO PRÁTICO - INTELIGÊNCIA ARTIFICIAL
# Aluno: Alessandro Augusto F. D. Oliveira
# Professor: Douglas Castilho
# Sistema de Análise de Vinho com Interface Gráfica e CLI
# ==============================================================================

import sys
import os
import tkinter as tk
from tkinter import messagebox

def launch_gui():
    """Importa e inicia a Interface Gráfica (GUI)."""
    print("Iniciando a Interface Gráfica...")
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont
    
    from gui.app import MainWindow

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

def launch_cli():
    """Importa e inicia a Interface de Linha de Comando (CLI) com bibliotecas gráficas."""
    print("Iniciando a Interface de Linha de Comando (com bibliotecas gráficas)...")
    from cli import menu_principal
    menu_principal()

def launch_cli_manual():
    """Importa e inicia a CLI com visualizações manuais, baseadas em texto."""
    print("Iniciando a Interface de Linha de Comando (versão com plots manuais)...")
    from cli_manual import menu_principal
    menu_principal()

def launch_cli_gui():
    """Importa e inicia a Interface Gráfica baseada em Tkinter (CLI melhorado)."""
    print("Iniciando a Interface Gráfica CLI...")
    from gui.cli_gui import WineQualityApp
    
    root = tk.Tk()
    app = WineQualityApp(root)
    
    def on_closing():
        if messagebox.askokcancel("Sair", "Deseja realmente sair do programa?"):
            root.destroy()
            os._exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    while True:
        print("\n" + "="*30)
        print("   SISTEMA DE ANÁLISE DE VINHOS")
        print("="*30)
        print("Escolha o modo de execução:")
        print("  1. Interface Gráfica (GUI)")
        print("  2. Interface de Terminal (CLI com Gráficos)")
        print("  3. Interface de Terminal (CLI Manual - Texto)")
        print("  4. Interface Gráfica (CLI)")
        print("  0. Sair")
        
        choice = input("Digite sua escolha: ")
        
        if choice == '1':
            launch_gui()
            break
        elif choice == '2':
            launch_cli()
            break
        elif choice == '3':
            launch_cli_manual()
            break
        elif choice == '4':
            launch_cli_gui()
            break
        elif choice == '0':
            break
        else:
            print("Opção inválida. Tente novamente.")