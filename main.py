import sys
import time
import psutil
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QTextEdit, QGridLayout,
                             QPushButton, QDateTimeEdit, QSizePolicy, QTextBrowser,
                             QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QDateTime, pyqtSlot
import pynvml  # NVIDIA library for GPU monitoring
from datetime import datetime
import sqlite3

# Import matplotlib components for embedding in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

# Alert threshold settings
GPU_TEMP_THRESHOLD = 80  # Trigger alert if GPU temp > 80°C
GPU_UTIL_THRESHOLD = 90  # Trigger alert if GPU utilization > 90%

# Database file for storing historical data
DATABASE_FILE = "resource_monitor.db"

class GPUResourceMonitor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Advanced Resource Monitor')
        self.setGeometry(100, 100, 900, 800)  # Adjusted window size to accommodate the plots

        # Initialize database
        self.init_database()

        # Apply the theme using stylesheets
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
            }
            QTabBar::tab {
                background: #2A2A2A;
                color: #CCCCCC;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background: #3A3A3A;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
            QTextEdit, QTextBrowser {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
            }
            QPushButton {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: none;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
            QDateTimeEdit {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
            }
            QCalendarWidget QWidget#qt_calendar_navigationbar {
                background-color: #2A2A2A;
            }
            QCalendarWidget QToolButton {
                background: transparent;
                border: none;
                color: #FFFFFF;
            }
            QCalendarWidget QToolButton:hover {
                background-color: #3A3A3A;
            }
            QCalendarWidget QSpinBox {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: none;
            }
            QCalendarWidget QAbstractItemView {
                selection-background-color: #6C5DD3;
                selection-color: #FFFFFF;
                background-color: #1E1E1E;
                color: #CCCCCC;
            }
        """)

        # Create the main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create a QTabWidget to hold multiple tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create the monitoring tab
        self.monitor_tab = QWidget()
        self.tabs.addTab(self.monitor_tab, "Monitoring")
        monitor_layout = QVBoxLayout()
        self.monitor_tab.setLayout(monitor_layout)

        # Create a grid layout for the labels and indicators
        info_layout = QGridLayout()
        monitor_layout.addLayout(info_layout)

        # Create labels and indicators
        self.gpu_label = QLabel("GPU Status: ")
        self.gpu_indicator = QLabel()
        self.gpu_indicator.setFixedSize(20, 20)
        self.gpu_indicator.setStyleSheet("background-color: green; border-radius: 10px;")

        self.cpu_label = QLabel("CPU Status: ")
        self.cpu_indicator = QLabel()
        self.cpu_indicator.setFixedSize(20, 20)
        self.cpu_indicator.setStyleSheet("background-color: green; border-radius: 10px;")

        self.ram_label = QLabel("RAM Status: ")
        self.ram_indicator = QLabel()
        self.ram_indicator.setFixedSize(20, 20)
        self.ram_indicator.setStyleSheet("background-color: green; border-radius: 10px;")

        # Add labels and indicators to the grid layout
        info_layout.addWidget(self.gpu_label, 0, 0)
        info_layout.addWidget(self.gpu_indicator, 0, 1)
        info_layout.addWidget(self.cpu_label, 1, 0)
        info_layout.addWidget(self.cpu_indicator, 1, 1)
        info_layout.addWidget(self.ram_label, 2, 0)
        info_layout.addWidget(self.ram_indicator, 2, 1)

        # Create a label for textual data
        self.label = QLabel("Initializing monitoring...", self)
        self.label.setAlignment(Qt.AlignCenter)
        monitor_layout.addWidget(self.label)

        # Create a matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        monitor_layout.addWidget(self.canvas)

        # Apply theme to matplotlib plots
        self.apply_plot_theme()

        # Create the notifications tab
        self.notifications_tab = QWidget()
        self.tabs.addTab(self.notifications_tab, "Notifications")
        notifications_layout = QVBoxLayout()
        self.notifications_tab.setLayout(notifications_layout)

        # Create a text area to display notifications
        self.notifications_text = QTextEdit()
        self.notifications_text.setReadOnly(True)
        notifications_layout.addWidget(self.notifications_text)

        # Create the historical data tab
        self.history_tab = QWidget()
        self.tabs.addTab(self.history_tab, "History")
        history_layout = QVBoxLayout()
        self.history_tab.setLayout(history_layout)

        # Controls for historical data
        controls_layout = QHBoxLayout()
        history_layout.addLayout(controls_layout)

        self.start_date_edit = QDateTimeEdit(QDateTime.currentDateTime().addDays(-1))
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_date_edit.setCalendarPopup(True)  # Enable calendar popup
        self.end_date_edit = QDateTimeEdit(QDateTime.currentDateTime())
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_date_edit.setCalendarPopup(True)  # Enable calendar popup
        self.refresh_button = QPushButton("Load Data")
        self.refresh_button.clicked.connect(self.load_historical_data)

        controls_layout.addWidget(QLabel("Start Date:"))
        controls_layout.addWidget(self.start_date_edit)
        controls_layout.addWidget(QLabel("End Date:"))
        controls_layout.addWidget(self.end_date_edit)
        controls_layout.addWidget(self.refresh_button)

        # Figure and canvas for historical data
        self.history_figure = Figure()
        self.history_canvas = FigureCanvas(self.history_figure)
        history_layout.addWidget(self.history_canvas)

        # Apply theme to historical plot
        self.apply_plot_theme_history()

        # Create the help tab
        self.help_tab = QWidget()
        self.tabs.addTab(self.help_tab, "Help")
        help_layout = QVBoxLayout()
        self.help_tab.setLayout(help_layout)

        # Add help content
        self.help_text = QTextBrowser()
        self.help_text.setOpenExternalLinks(True)
        self.help_text.setHtml(self.get_help_content())
        help_layout.addWidget(self.help_text)

        # Data lists for plotting
        self.gpu_usage_data = []
        self.cpu_usage_data = []
        self.ram_usage_data = []
        self.time_stamps = []
        self.start_time = time.time()

        # Set up timer to refresh data every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second

        # Animation for indicators (optional)
        self.init_animations()

    def init_database(self):
        """Initialize the SQLite database for storing historical data."""
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()
        # Create table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_usage (
                timestamp TEXT,
                gpu_util REAL,
                gpu_temp REAL,
                cpu_util REAL,
                ram_util REAL
            )
        """)
        self.conn.commit()

    def apply_plot_theme(self):
        plt.style.use('dark_background')
        self.figure.patch.set_facecolor('#1E1E1E')
        self.figure.patch.set_alpha(0)

    def apply_plot_theme_history(self):
        plt.style.use('dark_background')
        self.history_figure.patch.set_facecolor('#1E1E1E')
        self.history_figure.patch.set_alpha(0)

    def init_animations(self):
        pass

    def update_metrics(self):
        # Fetch GPU and CPU metrics
        gpu_metrics = self.get_gpu_metrics()
        cpu_metrics = self.get_cpu_metrics()

        # Update the display with fetched metrics
        self.label.setText(f"GPU: {gpu_metrics}\nCPU: {cpu_metrics}")

        # Update plotting data
        current_time = time.time() - self.start_time
        self.time_stamps.append(current_time)

        # GPU data
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        self.gpu_usage_data.append(gpu_util)

        # CPU data
        cpu_util = psutil.cpu_percent(interval=None)
        self.cpu_usage_data.append(cpu_util)

        # RAM data
        ram_util = psutil.virtual_memory().percent
        self.ram_usage_data.append(ram_util)

        # Store data in the database
        self.store_data(datetime.now(), gpu_util, gpu_temp, cpu_util, ram_util)

        # Update indicators
        self.update_indicators(gpu_util, gpu_temp, cpu_util, ram_util)

        # Keep only the last N data points to avoid memory issues
        max_points = 100  # Adjust as needed
        if len(self.time_stamps) > max_points:
            self.time_stamps = self.time_stamps[-max_points:]
            self.gpu_usage_data = self.gpu_usage_data[-max_points:]
            self.cpu_usage_data = self.cpu_usage_data[-max_points:]
            self.ram_usage_data = self.ram_usage_data[-max_points:]

        # Update the plot
        self.figure.clear()
        # Create subplots
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312)
        ax3 = self.figure.add_subplot(313)

        # Plot GPU utilization
        ax1.plot(self.time_stamps, self.gpu_usage_data, color='#6C5DD3')
        ax1.set_title('GPU Utilization (%)', color='#FFFFFF')
        ax1.set_ylabel('Utilization (%)', color='#FFFFFF')
        ax1.tick_params(axis='x', colors='#CCCCCC')
        ax1.tick_params(axis='y', colors='#CCCCCC')
        ax1.grid(True, color='#2A2A2A')

        # Plot CPU utilization
        ax2.plot(self.time_stamps, self.cpu_usage_data, color='#19B7A0')
        ax2.set_title('CPU Utilization (%)', color='#FFFFFF')
        ax2.set_ylabel('Utilization (%)', color='#FFFFFF')
        ax2.tick_params(axis='x', colors='#CCCCCC')
        ax2.tick_params(axis='y', colors='#CCCCCC')
        ax2.grid(True, color='#2A2A2A')

        # Plot RAM usage
        ax3.plot(self.time_stamps, self.ram_usage_data, color='#E95F5C')
        ax3.set_title('RAM Usage (%)', color='#FFFFFF')
        ax3.set_xlabel('Time (s)', color='#FFFFFF')
        ax3.set_ylabel('Usage (%)', color='#FFFFFF')
        ax3.tick_params(axis='x', colors='#CCCCCC')
        ax3.tick_params(axis='y', colors='#CCCCCC')
        ax3.grid(True, color='#2A2A2A')

        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

    def store_data(self, timestamp, gpu_util, gpu_temp, cpu_util, ram_util):
        """Store the collected data into the database."""
        self.cursor.execute("""
            INSERT INTO resource_usage (timestamp, gpu_util, gpu_temp, cpu_util, ram_util)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp.strftime('%Y-%m-%d %H:%M:%S'), gpu_util, gpu_temp, cpu_util, ram_util))
        self.conn.commit()

    def update_indicators(self, gpu_util, gpu_temp, cpu_util, ram_util):
        """Update the color indicators based on the current metrics."""
        # GPU Indicator
        if gpu_util > 90 or gpu_temp > 80:
            gpu_color = 'red'
        elif gpu_util > 70 or gpu_temp > 70:
            gpu_color = 'yellow'
        else:
            gpu_color = 'green'
        self.gpu_indicator.setStyleSheet(f"background-color: {gpu_color}; border-radius: 10px;")

        # CPU Indicator
        if cpu_util > 90:
            cpu_color = 'red'
        elif cpu_util > 70:
            cpu_color = 'yellow'
        else:
            cpu_color = 'green'
        self.cpu_indicator.setStyleSheet(f"background-color: {cpu_color}; border-radius: 10px;")

        # RAM Indicator
        if ram_util > 90:
            ram_color = 'red'
        elif ram_util > 70:
            ram_color = 'yellow'
        else:
            ram_color = 'green'
        self.ram_indicator.setStyleSheet(f"background-color: {ram_color}; border-radius: 10px;")

    def get_gpu_metrics(self):
        # Get GPU information using NVIDIA's nvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Check if any alerts need to be triggered
        if gpu_temp > GPU_TEMP_THRESHOLD or gpu_util.gpu > GPU_UTIL_THRESHOLD:
            self.add_notification(gpu_temp, gpu_util.gpu)

        return f"Memory Used: {mem_info.used / (1024 ** 2):.2f} MB / {mem_info.total / (1024 ** 2):.2f} MB, " \
               f"Temp: {gpu_temp} °C, Utilization: {gpu_util.gpu}%"

    def get_cpu_metrics(self):
        # Get CPU information using psutil
        cpu_util = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        return f"CPU Usage: {cpu_util}%, Memory Usage: {memory.percent}%"

    def add_notification(self, temp, util):
        # Add a notification message to the notifications tab
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{current_time}] Warning: GPU threshold exceeded!\n" \
                  f"GPU Temperature: {temp} °C\n" \
                  f"GPU Utilization: {util}%\n"

        # Append the message to the notifications text area
        self.notifications_text.append(message)
        print("Alert added to notifications tab.")

    @pyqtSlot()
    def load_historical_data(self):
        """Load historical data based on the selected date range."""
        start_time = self.start_date_edit.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end_time = self.end_date_edit.dateTime().toString("yyyy-MM-dd HH:mm:ss")

        # Fetch data from the database
        self.cursor.execute("""
            SELECT timestamp, gpu_util, cpu_util, ram_util
            FROM resource_usage
            WHERE timestamp BETWEEN ? AND ?
        """, (start_time, end_time))
        rows = self.cursor.fetchall()

        if not rows:
            QMessageBox.information(self, "No Data Available",
                                    "No data is available for the selected date range.")
            return

        timestamps = []
        gpu_data = []
        cpu_data = []
        ram_data = []

        # Check for missing data
        expected_count = int((self.end_date_edit.dateTime().toSecsSinceEpoch() -
                              self.start_date_edit.dateTime().toSecsSinceEpoch()) / 1)  # Assuming data every second
        actual_count = len(rows)

        if actual_count < expected_count:
            QMessageBox.warning(self, "Partial Data",
                                "Data is not available for some of the selected time range.")

        for row in rows:
            timestamp = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            timestamps.append(timestamp)
            gpu_data.append(row[1])
            cpu_data.append(row[2])
            ram_data.append(row[3])

        # Plot the historical data
        self.history_figure.clear()
        ax1 = self.history_figure.add_subplot(311)
        ax2 = self.history_figure.add_subplot(312)
        ax3 = self.history_figure.add_subplot(313)

        # GPU plot
        ax1.plot(timestamps, gpu_data, color='#6C5DD3')
        ax1.set_title('Historical GPU Utilization (%)', color='#FFFFFF')
        ax1.set_ylabel('Utilization (%)', color='#FFFFFF')
        ax1.tick_params(axis='x', colors='#CCCCCC', rotation=45)
        ax1.tick_params(axis='y', colors='#CCCCCC')
        ax1.grid(True, color='#2A2A2A')

        # CPU plot
        ax2.plot(timestamps, cpu_data, color='#19B7A0')
        ax2.set_title('Historical CPU Utilization (%)', color='#FFFFFF')
        ax2.set_ylabel('Utilization (%)', color='#FFFFFF')
        ax2.tick_params(axis='x', colors='#CCCCCC', rotation=45)
        ax2.tick_params(axis='y', colors='#CCCCCC')
        ax2.grid(True, color='#2A2A2A')

        # RAM plot
        ax3.plot(timestamps, ram_data, color='#E95F5C')
        ax3.set_title('Historical RAM Usage (%)', color='#FFFFFF')
        ax3.set_xlabel('Timestamp', color='#FFFFFF')
        ax3.set_ylabel('Usage (%)', color='#FFFFFF')
        ax3.tick_params(axis='x', colors='#CCCCCC', rotation=45)
        ax3.tick_params(axis='y', colors='#CCCCCC')
        ax3.grid(True, color='#2A2A2A')

        # Adjust layout
        self.history_figure.tight_layout()
        self.history_canvas.draw()

    def get_help_content(self):
        """Return the HTML content for the Help tab."""
        help_content = """
        <h2>How to Use the Advanced Resource Monitor</h2>
        <p>This application provides real-time monitoring of your system's GPU, CPU, and RAM usage. It also allows you to view historical data and receive notifications when certain thresholds are exceeded.</p>

        <h3>Monitoring Tab</h3>
        <p>The <strong>Monitoring</strong> tab displays real-time graphs for GPU, CPU, and RAM usage. It also shows status indicators next to each component:</p>
        <ul>
            <li><span style="color:green;">Green</span>: Excellent performance.</li>
            <li><span style="color:yellow;">Yellow</span>: Moderate usage or temperatures.</li>
            <li><span style="color:red;">Red</span>: High usage or temperatures; attention needed.</li>
        </ul>

        <h3>Notifications Tab</h3>
        <p>The <strong>Notifications</strong> tab displays alert messages when the GPU temperature or utilization exceeds predefined thresholds.</p>

        <h3>History Tab</h3>
        <p>The <strong>History</strong> tab allows you to view historical performance data over a selected date range. Use the date-time selectors to choose the start and end dates. Click on the date fields to open a calendar widget for easy date selection. After selecting the dates, click <strong>Load Data</strong> to display the graphs.</p>
        <p>If data is not available for some or all of the selected time range, a message will inform you accordingly.</p>

        <h3>Help Tab</h3>
        <p>You are currently viewing the <strong>Help</strong> tab. Here, you can find information on how to use the application and its features.</p>

        <h3>Additional Information</h3>
        <p>Data is stored locally in a database file named <code>resource_monitor.db</code> in the same directory as this application.</p>
        <p>For questions or support, please contact me.</p>
        """
        return help_content

    def closeEvent(self, event):
        """Close the database connection when the application is closed."""
        self.conn.close()
        event.accept()

def run_gui_app():
    app = QApplication(sys.argv)
    monitor = GPUResourceMonitor()
    monitor.show()
    sys.exit(app.exec_())

# Start the GUI application
if __name__ == '__main__':
    run_gui_app()
