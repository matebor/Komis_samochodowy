
#Projekt Mateusza BORYSIAKA
#Informatyka w Biznesie, II stopień
#Kurs: Usługi Sieciowe

from tkinter import*
import tkinter.messagebox
import mysql.connector
import numpy as np
from numpy.core.fromnumeric import size 
import pandas as pd
import matplotlib.pyplot as plt


class OknoLogowanie:

    def __init__(self, master):
        self.master = master
        self.master.title("Logowanie - Aplikacja komis samochodów")
        self.master.geometry('480x480')
        self.master.iconbitmap(r'C:\Users\Mateusz\Desktop\Projekt_ver1\user.ico')

        self.Label = Label(self.master, text = 'Aplikacja komis samochodowy')
        self.Label.place(x=150, y=20)

        self.Login = StringVar()
        self.Haslo = StringVar()

        #Etykiety
        self.LabelLogin = Label(self.master, text = 'Login:')
        self.LabelLogin.place(x=140, y=200)

        self.LabelHaslo = Label(self.master, text = 'Hasło:')
        self.LabelHaslo.place(x=140, y=230)

        #Miejsca do wrowadzania danych
        self.BoxLogin = Entry(self.master, width=25, bg='white', textvariable= self.Login)
        self.BoxLogin.place(x=190, y=200)

        self.BoxHaslo = Entry(self.master, width=25, bg='white', show='*', textvariable= self.Haslo)
        self.BoxHaslo.place(x=190, y=230)

        #Przyciski
        self.PrzycikZaloguj = Button(self.master, text = 'Zaloguj', width=25, bg='white', command=lambda:[self.PokazLogin(), self.Login_System()])
        self.PrzycikZaloguj.place(x=150, y=300)

        self.PrzyciskWyczysc = Button(self.master, text = 'Wyczyść',  width=25, bg='white', command=self.Reset)
        self.PrzyciskWyczysc.place(x=150, y=330)

        self.PrzyciskZamknij = Button(self.master, text = 'Zamknij aplikajcę',  width=25, bg='white', command=self.Wyjscie)
        self.PrzyciskZamknij.place(x=150, y=360)
    
    
    def PokazLogin(self):
        global varLogin
        varLogin = self.Login.get()


    def Login_System(self):
        
        u = self.Login.get()
        p = self.Haslo.get()
        
        
        connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database='uzytkownicy',
        auth_plugin='mysql_native_passowrd'
        )

        zapytanieLogin = (f"SELECT * FROM uzytkownicy WHERE login='{u}'")
        zapytanieHaslo = (f"SELECT hasło FROM uzytkownicy WHERE login='{u}'")

        mojKursor = connection.cursor()
        mojKursor.execute(zapytanieLogin)

        wynik = mojKursor.fetchone()

        if (mojKursor.rowcount != 0):
    
            mojKursor.execute(zapytanieHaslo)
            wynik2 = mojKursor.fetchone()

            for row in wynik2:
                if (p == row):
                    self.noweOkno = Toplevel(self.master)
                    self.app = OknoPanel(self.noweOkno)
                else:
                    tkinter.messagebox.askyesno("Błąd", "Błędne hasło")
                
        else:
            tkinter.messagebox.askyesno("Błąd", "Wprowadzono zły login lub hasło")
            self.Login.set("")
            self.Haslo.set("")
            self.BoxLogin.focus()

        connection.close()
        

    def Reset(self):
        self.Login.set("")
        self.Haslo.set("")
        self.BoxLogin.focus()


    def Wyjscie(self):
        self.Wyjscie = tkinter.messagebox.askyesno("Wyjście", "Czy chcesz opuścić program?")
        if self.Wyjscie > 0:
            self.master.destroy()
        else:
            command = self.nowe_okno
            return


    def nowe_okno(self):
        self.noweOkno = Toplevel(self.master)
        self.app = OknoPanel(self.noweOkno)


class OknoPanel(OknoLogowanie):

    def __init__(self, master):
        self.master = master
        self.master.title("Panel - Aplikacja komis samochodów")
        self.master.geometry('730x480')
        self.master.iconbitmap(r'C:\Users\Mateusz\Desktop\Projekt_ver1\user.ico')

        #Etykieta zalogowany uzytkownik
        self.LabelZalogowany = Label(self.master, text = 'Zalogowany użytkownik: ')
        self.LabelZalogowany.place(x=10, y=5)
        self.LabelZalogowany2 = Label(self.master, text = varLogin)
        self.LabelZalogowany2.place(x=150, y=5)

        self.Marka = StringVar()
        self.Model = StringVar()
        self.Paliwo = StringVar()
        self.Dzwi = StringVar()
        self.Nadwozie = StringVar()
        self.Naped = StringVar()
        self.Waga = StringVar()
        self.Cylindry = StringVar()
        self.Silnik = StringVar()
        self.Moc = StringVar()
        self.Moment = StringVar()
        self.Cena = StringVar()

        #Etykiety i miejsce do wprowadzania danych
        self.LabelMarka = Label(self.master, text = 'Marka: ')
        self.LabelMarka.place(x=10, y=40)
        self.BoxMarka = Entry(self.master, width=25, bg='white', textvariable= self.Marka)
        self.BoxMarka.place(x=120, y=40)

        self.LabelModel = Label(self.master, text = 'Model: ')
        self.LabelModel.place(x=10, y=70)
        self.BoxModel = Entry(self.master, width=25, bg='white', textvariable= self.Model)
        self.BoxModel.place(x=120, y=70)

        self.LabelPaliwo = Label(self.master, text = 'Typ paliwa: ')
        self.LabelPaliwo.place(x=10, y=100)
        self.BoxPaliwo = Entry(self.master, width=25, bg='white', textvariable= self.Paliwo)
        self.BoxPaliwo.place(x=120, y=100)

        self.LabelDzwi = Label(self.master, text = 'Ilość dzwi: ')
        self.LabelDzwi.place(x=10, y=130)
        self.BoxDzwi = Entry(self.master, width=25, bg='white', textvariable= self.Dzwi)
        self.BoxDzwi.place(x=120, y=130)

        self.LabelNadwozie = Label(self.master, text = 'Typ nadwozia: ')
        self.LabelNadwozie.place(x=10, y=160)
        self.BoxNadwozie = Entry(self.master, width=25, bg='white', textvariable= self.Nadwozie)
        self.BoxNadwozie.place(x=120, y=160)

        self.LabelNaped = Label(self.master, text = 'Typ napędu: ')
        self.LabelNaped.place(x=10, y=190)
        self.BoxNaped = Entry(self.master, width=25, bg='white', textvariable= self.Naped)
        self.BoxNaped.place(x=120, y=190)

        self.LabelWaga = Label(self.master, text = 'Waga: ')
        self.LabelWaga.place(x=10, y=220)
        self.BoxWaga = Entry(self.master, width=25, bg='white', textvariable= self.Waga)
        self.BoxWaga.place(x=120, y=220)

        self.LabelCylindry = Label(self.master, text = 'Ilość cylindrów: ')
        self.LabelCylindry.place(x=10, y=250)
        self.BoxCylindry = Entry(self.master, width=25, bg='white', textvariable= self.Cylindry)
        self.BoxCylindry.place(x=120, y=250)

        self.LabelSilnik = Label(self.master, text = 'Pojemność silnika: ')
        self.LabelSilnik.place(x=10, y=280)
        self.BoxSilnik = Entry(self.master, width=25, bg='white', textvariable= self.Silnik)
        self.BoxSilnik.place(x=120, y=280)

        self.LabelMoc = Label(self.master, text = 'Moc silnika: ')
        self.LabelMoc.place(x=10, y=310)
        self.BoxMoc = Entry(self.master, width=25, bg='white', textvariable= self.Moc )
        self.BoxMoc.place(x=120, y=310)

        self.LabelMoment = Label(self.master, text = 'Moment obrotowy: ')
        self.LabelMoment.place(x=10, y=340)
        self.BoxMoment = Entry(self.master, width=25, bg='white', textvariable= self.Moment)
        self.BoxMoment.place(x=120, y=340)

        self.LabelCena = Label(self.master, text = 'Prognozowana cena: ')
        self.LabelCena.place(x=300, y=40)
        self.BoxCena = Entry(self.master, width=25, bg='yellow', textvariable= self.Cena)
        self.BoxCena.place(x=430, y=40)

        #self.LabelDane = Label(self.master, text = 'Ostatni zapis: ')
        #self.LabelDane.place(x=300, y=100)

        #Przyciski
        self.PrzyciskWyczyscPanel = Button(self.master, text = 'Wyczyść',  width=20, bg='white', command=self.Reset2)
        self.PrzyciskWyczyscPanel.place(x=120, y=380)

        self.PrzyciskOblicz = Button(self.master, text = 'Oblicz cenę',  width=20, bg='white', command=self.Obliczanie)
        self.PrzyciskOblicz.place(x=430, y=65)

        self.PrzyciskDodaj = Button(self.master, text = 'Dodaj do BD',  width=20, bg='white', command=self.Dodaj)
        self.PrzyciskDodaj.place(x=300, y=380)

        self.PrzyciskZamknij = Button(self.master, text = 'Wyloguj',  width=20, bg='white', command=self.Wylogowanie)
        self.PrzyciskZamknij.place(x=565, y=380)

        #self.PrzyciskWczytaj = Button(self.master, text = 'Wczytaj',  width=20, bg='white', command=self.Wczytaj)
        #self.PrzyciskWczytaj.place(x=430, y=100)


    def Reset2(self):
        self.Marka.set("")
        self.Model.set("")
        self.Paliwo.set("")
        self.Dzwi.set("")
        self.Nadwozie.set("")
        self.Naped.set("")
        self.Waga.set("")
        self.Cylindry.set("")
        self.Silnik.set("")
        self.Moc.set("")
        self.Moment.set("")
        self.BoxMarka.focus()


    def Obliczanie(self):
        # Zastosowanie AI - sieci neuronowe
        from tensorflow.keras.utils import to_categorical
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.python.keras.models import Model

        input_file = "ceny.csv"
        pd.options.display.max_columns = None
        main_df = pd.read_csv(input_file)
        main_df
        
        Marka = self.Marka.get()
        Model = self.Model.get()
        Typ_paliwa = self.Paliwo.get()
        Liczba_dzwi = self.Dzwi.get()
        Nadwozie = self.Nadwozie.get()
        Naped = self.Naped.get()
        Waga = self.Waga.get()
        Liczba_cylindrow = self.Cylindry.get()
        Pojemnosc_silnika = self.Silnik.get()
        Moc = self.Moc.get()
        Moment_obrotowy = self.Moment.get()
        Cena = float(15000)
       
        NowyWiersz = {'ID_auto': '206', 'Marka': Marka, 'Model': Model, 'Typ_paliwa': Typ_paliwa, 'Liczba_dzwi': Liczba_dzwi, 'Nadwozie': Nadwozie, 'Naped': Naped, 'Waga': Waga, 'Liczba_cylindrow': Liczba_cylindrow, 'Pojemnosc_silnika': Pojemnosc_silnika, 'Moc': Moc, 'Moment_obrotowy': Moment_obrotowy, 'Cena': Cena}
        main_df = main_df.append(NowyWiersz, ignore_index=True)

        main_df.drop(columns=['Model'], inplace=True)
        main_df

        encoding_columns = ['Marka', 'Typ_paliwa', 'Liczba_dzwi', 'Nadwozie', 'Naped', 'Liczba_cylindrow', 'Pojemnosc_silnika']
        main_array = np.array(main_df.ID_auto).reshape(-1, 1)
        for column in main_df.columns:
            if column in encoding_columns:
                temp = np.array(pd.get_dummies(main_df[column]))
            else:
                temp = np.array(main_df[column]).reshape(-1, 1)
            main_array = np.hstack((main_array, temp)) 
        main_array = main_array[:, 2:]  
        pd.DataFrame(main_array)

        X_data = main_array[:, :-1]
        y_data = main_array[:, -1].reshape(-1, 1)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_data_scaled = x_scaler.fit_transform(X_data)
        y_data_scaled = y_scaler.fit_transform(y_data)
        print("Shape of X_data: {}".format(X_data.shape))
        print("Shape of y_data: {}".format(y_data.shape))
        print("==========X_data after rescaling===============")
        print(pd.DataFrame(X_data_scaled).head())
        print("==========y_data after rescaling===============")
        print(y_data_scaled.ravel())

        X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data_scaled, test_size=0.1, shuffle=False)
        print("Shape of X_train: {}".format(X_train.shape))
        print("Shape of X_test: {}".format(X_test.shape))
        print("Shape of y_train: {}".format(y_train.shape))
        print("Shape of y_test: {}".format(y_test.shape))

        model = Sequential()
        model.add(Dense(8, activation='relu', input_shape=(None, len(main_array[0])-1)))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()

        history = model.fit(x=X_train, y=y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

        PREDICT_ROW = 205
        predict_data = main_array[PREDICT_ROW, :].reshape(1, -1)
        X_predict = predict_data[:, :-1]
        y_true = predict_data[:, -1]
        predict_data_scaled = x_scaler.transform(X_predict)
        y_pred_scaled = model.predict(predict_data_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        print("\n")
        print("Wynik prognozy ceny: {}".format(float(y_pred)))
        print("Prawdziwa cena: {}".format(float(y_true)))
        print("Procent błędu: {}".format(str(float(abs(y_true - y_pred) * 100 / y_true))))
        if (y_pred < 0):
            self.Obliczanie()
        else:
            PrzewidywanaCena = format(int(y_pred))

            self.Cena.set(PrzewidywanaCena)
            self.BoxCena.focus()
        
    def Wylogowanie(self):
        self.Wylogowanie = tkinter.messagebox.askyesno("Wylogowanie", "Czy chcesz się wylogować?")
        if self.Wylogowanie > 0:
            self.master.destroy()
        else:
            return
    
    def Dodaj(self):
        
        connection2 = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database='uzytkownicy',
        auth_plugin='mysql_native_passowrd'
        )

        ZapytanieLiczbaWierszy = (f"SELECT COUNT(*) FROM samochody")
        mojKursor3 = connection2.cursor()
        mojKursor3.execute(ZapytanieLiczbaWierszy)
        Wynik3 = mojKursor3.fetchone()

        LiczbaWierszy = Wynik3[0]+1

        ZapytanieOperator = (f"SELECT ID_uzytkownik FROM uzytkownicy WHERE login='{varLogin}'")
        mojKursor4 = connection2.cursor()
        mojKursor4.execute(ZapytanieOperator)
        Wynik4 = mojKursor4.fetchone()

        for ID_operator in Wynik4:
            print(ID_operator)
        
        ID_Auto = LiczbaWierszy
        Marka = self.Marka.get()
        Model = self.Model.get()
        Typ_paliwa = self.Paliwo.get()
        Liczba_dzwi = self.Dzwi.get()
        Nadwozie = self.Nadwozie.get()
        Naped = self.Naped.get()
        Waga = self.Waga.get()
        Liczba_cylindrow = self.Cylindry.get()
        Pojemnosc_silnika = self.Silnik.get()
        Moc = self.Moc.get()
        Moment_obrotowy = self.Moment.get()
        Cena = self.Cena.get()
        Operator = ID_operator

        
        ZapytanieWstaw = (f"INSERT INTO samochody(ID_auto, Marka, Model, Typ_paliwa, Liczba_dzwi, Nadwozie, Naped, Waga, Liczba_cylindrow, Pojemnosc_silnika, Moc, Moment_obrotowy, Cena, Operator) VALUES ({ID_Auto},'{Marka}','{Model}','{Typ_paliwa}',{Liczba_dzwi},'{Nadwozie}','{Naped}',{Waga},{Liczba_cylindrow},{Pojemnosc_silnika},{Moc},{Moment_obrotowy},{Cena},{Operator})")

        mojKursor2 = connection2.cursor()
        mojKursor2.execute(ZapytanieWstaw)

        connection2.commit()
        connection2.close()
        
        
        self.LabelDodaj = Label(self.master, text = 'Dane zostały dodane', bg="green", fg="white", width=20)
        self.LabelDodaj.place(x=300, y=340)


root = Tk()
app = OknoLogowanie(root)
root.mainloop()