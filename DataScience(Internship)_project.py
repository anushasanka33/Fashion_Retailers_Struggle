{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOf6/B8zaP2Sdu8/3EChnGi",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/anushasanka33/Fashion_Retailers_Struggle/blob/main/DataScience(Internship)_project.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**PROBLEM STATEMENT**"
      ],
      "metadata": {
        "id": "SBWbImuWRKND"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Fashion retailers struggle to align inventory levels with unpredictable consumer demands, causing overstocking, markdowns, and environmental waste, highlighting the need for a data science-driven predictive analytics model to enhance inventory management and promote sustainability.**"
      ],
      "metadata": {
        "id": "Vl4s6KNKRYFe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SOURCE CODE**"
      ],
      "metadata": {
        "id": "F1OfWs50RjIc"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "N3RjE7DU8k5r"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**READING DATASET**"
      ],
      "metadata": {
        "id": "zcC596d5yq2D"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"/content/retail_sales_dataset.csv\")\n",
        "df"
      ],
      "metadata": {
        "id": "RsO8plMP9ZgF",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "outputId": "f796fd45-ec76-4c2e-a1c3-8cf073170cfe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     Transaction ID        Date Customer ID  Gender  Age Product Category  \\\n",
              "0                 1  2023-11-24     CUST001    Male   34           Beauty   \n",
              "1                 2  2023-02-27     CUST002  Female   26         Clothing   \n",
              "2                 3  2023-01-13     CUST003    Male   50      Electronics   \n",
              "3                 4  2023-05-21     CUST004    Male   37         Clothing   \n",
              "4                 5  2023-05-06     CUST005    Male   30           Beauty   \n",
              "..              ...         ...         ...     ...  ...              ...   \n",
              "995             996  2023-05-16     CUST996    Male   62         Clothing   \n",
              "996             997  2023-11-17     CUST997    Male   52           Beauty   \n",
              "997             998  2023-10-29     CUST998  Female   23           Beauty   \n",
              "998             999  2023-12-05     CUST999  Female   36      Electronics   \n",
              "999            1000  2023-04-12    CUST1000    Male   47      Electronics   \n",
              "\n",
              "     Quantity  Price per Unit  Total Amount  \n",
              "0           3              50           150  \n",
              "1           2             500          1000  \n",
              "2           1              30            30  \n",
              "3           1             500           500  \n",
              "4           2              50           100  \n",
              "..        ...             ...           ...  \n",
              "995         1              50            50  \n",
              "996         3              30            90  \n",
              "997         4              25           100  \n",
              "998         3              50           150  \n",
              "999         4              30           120  \n",
              "\n",
              "[1000 rows x 9 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-4cc736f3-1037-47c6-8be0-54a08d049c42\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Transaction ID</th>\n",
              "      <th>Date</th>\n",
              "      <th>Customer ID</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Age</th>\n",
              "      <th>Product Category</th>\n",
              "      <th>Quantity</th>\n",
              "      <th>Price per Unit</th>\n",
              "      <th>Total Amount</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>2023-11-24</td>\n",
              "      <td>CUST001</td>\n",
              "      <td>Male</td>\n",
              "      <td>34</td>\n",
              "      <td>Beauty</td>\n",
              "      <td>3</td>\n",
              "      <td>50</td>\n",
              "      <td>150</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>2023-02-27</td>\n",
              "      <td>CUST002</td>\n",
              "      <td>Female</td>\n",
              "      <td>26</td>\n",
              "      <td>Clothing</td>\n",
              "      <td>2</td>\n",
              "      <td>500</td>\n",
              "      <td>1000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>2023-01-13</td>\n",
              "      <td>CUST003</td>\n",
              "      <td>Male</td>\n",
              "      <td>50</td>\n",
              "      <td>Electronics</td>\n",
              "      <td>1</td>\n",
              "      <td>30</td>\n",
              "      <td>30</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>2023-05-21</td>\n",
              "      <td>CUST004</td>\n",
              "      <td>Male</td>\n",
              "      <td>37</td>\n",
              "      <td>Clothing</td>\n",
              "      <td>1</td>\n",
              "      <td>500</td>\n",
              "      <td>500</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>2023-05-06</td>\n",
              "      <td>CUST005</td>\n",
              "      <td>Male</td>\n",
              "      <td>30</td>\n",
              "      <td>Beauty</td>\n",
              "      <td>2</td>\n",
              "      <td>50</td>\n",
              "      <td>100</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>995</th>\n",
              "      <td>996</td>\n",
              "      <td>2023-05-16</td>\n",
              "      <td>CUST996</td>\n",
              "      <td>Male</td>\n",
              "      <td>62</td>\n",
              "      <td>Clothing</td>\n",
              "      <td>1</td>\n",
              "      <td>50</td>\n",
              "      <td>50</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>996</th>\n",
              "      <td>997</td>\n",
              "      <td>2023-11-17</td>\n",
              "      <td>CUST997</td>\n",
              "      <td>Male</td>\n",
              "      <td>52</td>\n",
              "      <td>Beauty</td>\n",
              "      <td>3</td>\n",
              "      <td>30</td>\n",
              "      <td>90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>997</th>\n",
              "      <td>998</td>\n",
              "      <td>2023-10-29</td>\n",
              "      <td>CUST998</td>\n",
              "      <td>Female</td>\n",
              "      <td>23</td>\n",
              "      <td>Beauty</td>\n",
              "      <td>4</td>\n",
              "      <td>25</td>\n",
              "      <td>100</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>998</th>\n",
              "      <td>999</td>\n",
              "      <td>2023-12-05</td>\n",
              "      <td>CUST999</td>\n",
              "      <td>Female</td>\n",
              "      <td>36</td>\n",
              "      <td>Electronics</td>\n",
              "      <td>3</td>\n",
              "      <td>50</td>\n",
              "      <td>150</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>999</th>\n",
              "      <td>1000</td>\n",
              "      <td>2023-04-12</td>\n",
              "      <td>CUST1000</td>\n",
              "      <td>Male</td>\n",
              "      <td>47</td>\n",
              "      <td>Electronics</td>\n",
              "      <td>4</td>\n",
              "      <td>30</td>\n",
              "      <td>120</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1000 rows × 9 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-4cc736f3-1037-47c6-8be0-54a08d049c42')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-4cc736f3-1037-47c6-8be0-54a08d049c42 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-4cc736f3-1037-47c6-8be0-54a08d049c42');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-fdbeef6c-bc87-4671-ac09-f7c2404ce4ee\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-fdbeef6c-bc87-4671-ac09-f7c2404ce4ee')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-fdbeef6c-bc87-4671-ac09-f7c2404ce4ee button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**FEATURE ENGINEERING**"
      ],
      "metadata": {
        "id": "CxhENhzYyv0E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.dtypes"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MnBBYh_p9uI1",
        "outputId": "096c0861-1680-4c51-d160-7cdcacc9b0ce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Transaction ID       int64\n",
              "Date                object\n",
              "Customer ID         object\n",
              "Gender              object\n",
              "Age                  int64\n",
              "Product Category    object\n",
              "Quantity             int64\n",
              "Price per Unit       int64\n",
              "Total Amount         int64\n",
              "dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y9ko9P8a97_k",
        "outputId": "80dca682-3aca-40a1-ed33-0b5bd7e05b8a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1000 entries, 0 to 999\n",
            "Data columns (total 9 columns):\n",
            " #   Column            Non-Null Count  Dtype \n",
            "---  ------            --------------  ----- \n",
            " 0   Transaction ID    1000 non-null   int64 \n",
            " 1   Date              1000 non-null   object\n",
            " 2   Customer ID       1000 non-null   object\n",
            " 3   Gender            1000 non-null   object\n",
            " 4   Age               1000 non-null   int64 \n",
            " 5   Product Category  1000 non-null   object\n",
            " 6   Quantity          1000 non-null   int64 \n",
            " 7   Price per Unit    1000 non-null   int64 \n",
            " 8   Total Amount      1000 non-null   int64 \n",
            "dtypes: int64(5), object(4)\n",
            "memory usage: 70.4+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jw89SrSl-DgT",
        "outputId": "62ae56d7-0797-45f2-b75a-054d35fa56e7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Transaction ID      0\n",
              "Date                0\n",
              "Customer ID         0\n",
              "Gender              0\n",
              "Age                 0\n",
              "Product Category    0\n",
              "Quantity            0\n",
              "Price per Unit      0\n",
              "Total Amount        0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 82
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for i in df:\n",
        "    print('-------------')\n",
        "    print(df[i].value_counts().head(5))\n",
        "    print('-------------')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cYdwLnSo-HjO",
        "outputId": "f12f59bd-fc10-4432-ff74-f4039b194c08"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-------------\n",
            "1      1\n",
            "672    1\n",
            "659    1\n",
            "660    1\n",
            "661    1\n",
            "Name: Transaction ID, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "2023-05-16    11\n",
            "2023-07-14    10\n",
            "2023-05-23     9\n",
            "2023-08-05     8\n",
            "2023-02-05     8\n",
            "Name: Date, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "CUST001    1\n",
            "CUST672    1\n",
            "CUST659    1\n",
            "CUST660    1\n",
            "CUST661    1\n",
            "Name: Customer ID, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "Female    510\n",
            "Male      490\n",
            "Name: Gender, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "43    31\n",
            "64    31\n",
            "57    30\n",
            "51    30\n",
            "34    28\n",
            "Name: Age, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "Clothing       351\n",
            "Electronics    342\n",
            "Beauty         307\n",
            "Name: Product Category, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "4    263\n",
            "1    253\n",
            "2    243\n",
            "3    241\n",
            "Name: Quantity, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "50     211\n",
            "25     210\n",
            "500    199\n",
            "300    197\n",
            "30     183\n",
            "Name: Price per Unit, dtype: int64\n",
            "-------------\n",
            "-------------\n",
            "50      115\n",
            "100     108\n",
            "900      62\n",
            "200      62\n",
            "1200     54\n",
            "Name: Total Amount, dtype: int64\n",
            "-------------\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "# Assuming your dataset is stored in a DataFrame called 'df'\n",
        "# Selecting relevant features for the model\n",
        "features = ['Gender', 'Age', 'Product Category', 'Price per Unit']\n",
        "print(df)\n",
        "# Creating dummy variables for categorical features\n",
        "df_model = pd.get_dummies(df[features], columns=['Gender', 'Product Category'], drop_first=True, prefix=['Gender', 'Product'])\n",
        "\n",
        "# Adding the 'Age' and 'Price per Unit' columns\n",
        "df_model[['Age', 'Price per Unit']] = df[['Age', 'Price per Unit']]\n",
        "\n",
        "# Splitting the dataset into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(df_model, df['Quantity'], test_size=0.2, random_state=42)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_lkTBpvB0bek",
        "outputId": "0875e18d-4f17-44fc-b860-0938dfd3ffa5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     Transaction ID        Date Customer ID  Gender  Age Product Category  \\\n",
            "0                 1  2023-11-24     CUST001    Male   34           Beauty   \n",
            "1                 2  2023-02-27     CUST002  Female   26         Clothing   \n",
            "2                 3  2023-01-13     CUST003    Male   50      Electronics   \n",
            "3                 4  2023-05-21     CUST004    Male   37         Clothing   \n",
            "4                 5  2023-05-06     CUST005    Male   30           Beauty   \n",
            "..              ...         ...         ...     ...  ...              ...   \n",
            "995             996  2023-05-16     CUST996    Male   62         Clothing   \n",
            "996             997  2023-11-17     CUST997    Male   52           Beauty   \n",
            "997             998  2023-10-29     CUST998  Female   23           Beauty   \n",
            "998             999  2023-12-05     CUST999  Female   36      Electronics   \n",
            "999            1000  2023-04-12    CUST1000    Male   47      Electronics   \n",
            "\n",
            "     Quantity  Price per Unit  Total Amount  \n",
            "0           3              50           150  \n",
            "1           2             500          1000  \n",
            "2           1              30            30  \n",
            "3           1             500           500  \n",
            "4           2              50           100  \n",
            "..        ...             ...           ...  \n",
            "995         1              50            50  \n",
            "996         3              30            90  \n",
            "997         4              25           100  \n",
            "998         3              50           150  \n",
            "999         4              30           120  \n",
            "\n",
            "[1000 rows x 9 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Check `Product Category` field : only 3 attributes.\n",
        "Product_Category_List = df['Product Category'].unique()\n",
        "print(Product_Category_List)\n",
        "# Find the most popular Product Category in each male and female group : I like 'Clothing', too\n",
        "male_favorite_category =df[df['Gender'] == 'Male']['Product Category'].mode().values[0]\n",
        "female_favorite_category = df[df['Gender'] == 'Female']['Product Category'].mode().values[0]\n",
        "print(f\"[Men] Most Popular Product Category: {male_favorite_category}\")\n",
        "print(f\"[Women] Most Popular Product Category: {female_favorite_category}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DkGgJM360ia-",
        "outputId": "22eae114-2abe-4d46-a92a-25ea7722fb48"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['Beauty' 'Clothing' 'Electronics']\n",
            "[Men] Most Popular Product Category: Clothing\n",
            "[Women] Most Popular Product Category: Clothing\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**DATA VISUALIZATION**"
      ],
      "metadata": {
        "id": "YYIix0SmyfpN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "age_bins = [0, 18, 25, 35, 50, 100]\n",
        "age_labels = ['0-18', '19-25', '26-35', '36-50', '51+']\n",
        "df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)"
      ],
      "metadata": {
        "id": "GHNEJW3_-RqG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Now, you can create a bar plot to compare age groups and product categories\n",
        "plt.figure(figsize=(12, 6))\n",
        "sns.barplot(x='Age Group', y='Quantity', hue='Product Category', data=df)\n",
        "\n",
        "# You can customize the plot further\n",
        "plt.title('Product Category Purchase by Age Group')\n",
        "plt.xlabel('Age Group')\n",
        "plt.ylabel('Quantity Purchased')\n",
        "plt.xticks(rotation=45)  # Rotate x-axis labels for better readability\n",
        "\n",
        "# Show the plot\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 599
        },
        "id": "JWjSMyoD-rDJ",
        "outputId": "af51d134-dfa2-4d4b-e048-d726ba64c747"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1200x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA/MAAAJGCAYAAAAXjNXRAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACa9UlEQVR4nOzdeXxN1/7/8ffJJAmSSIkhpqCJMcYiNauZihprSKm5REvrXuG2pa1bqnVdxFxKY76qKorW0NKvVltVdKBqKEGFi0Qikens3x9+OddpDMlxIjnyej4eHj1n7bXX/uxzVtJ89lp7bZNhGIYAAAAAAIDDcMrrAAAAAAAAQM6QzAMAAAAA4GBI5gEAAAAAcDAk8wAAAAAAOBiSeQAAAAAAHAzJPAAAAAAADoZkHgAAAAAAB0MyDwAAAACAgyGZBwAAAADAwZDMAwByJCgoSHPnzs3rMPCIOnfunIKCgrR06dK8DiXbvv32WwUFBWn79u15HQoAoABxyesAAADZt3HjRk2cONHy3s3NTWXKlFGTJk00atQoFS9ePA+js49Vq1bJw8ND3bt3z/Y+KSkpWrNmjT799FOdOnVKqampls8lLCxMAQEBOYrh4MGD2rdvnwYOHCgvL6+cnoLDCQoKsrw2mUwqXry4AgMDNWLECDVq1CgPI8PdzJgxQ0uXLlXHjh3173//O6/DUWJioqKiorRjxw6dOXNGKSkpKlGihGrXrq1u3bqpZcuWeR0iADxySOYBwAG9+OKLKlu2rFJTU/XDDz9ozZo12rNnj7Zs2SIPD4+8Du+BrFmzRsWKFct2Mn/16lUNHTpUv/zyi1q1aqUuXbrI09NTp0+f1tatW7V+/Xr9/PPPOYrhxx9/VGRkpJ555pkCkcxLUpMmTRQaGirDMHTu3DmtWbNGAwcO1KJFi9SiRYu8Dg+3MQxDn376qfz9/fXFF18oMTFRRYoUybN4zpw5oyFDhujChQtq06aNunXrJk9PT128eFF79uzRiBEj9M4776hbt255FiMAPIpI5gHAATVv3ly1atWSJPXq1Us+Pj764IMPtGvXLnXp0uWO+yQlJcnT0/NhhvlQTJw4UUePHtWcOXPUvn17q21jx47VrFmz8iiy3Jeeni6z2Sw3N7cHbqtixYoKDQ21vG/btq26du2qDz/80C7JfHJyssNfaMovvv32W128eFErVqzQ0KFDtWPHDj3zzDN5Ekt6errCw8N15coVRUVFqX79+lbbw8PD9X//93/KyMi4ZzuP6u8nAMhN3DMPAI+Axo0bS7p1v7EkRUREqG7dujp79qyGDRumunXravz48ZJu/dE8ffp0tWjRQjVr1lT79u21dOlSGYZh1WZqaqrefvttNW7cWHXr1tXIkSN18eLFLMeOiIhQ69ats5TPnTvXavp2pk8++UQ9e/ZU7dq19cQTT6h///76v//7P0lS69at9fvvv+u7775TUFCQgoKCFBYWdtfzPnz4sL788kv17NkzSyIv3boNYcKECZb3x44dU0REhJ566inVqlVLTZo00cSJE3Xt2jWruGfMmCFJeuqppyxxZH62mefQvXt3BQcHq2HDhho3bpz+/PPPLMdftWqVnnrqKQUHB6tnz546cOCAwsLCspzTlStXNGnSJD355JOqVauWunbtqo8//tiqzu33ki9fvlxt2rRRrVq1dOTIEdWpU0dTp07NcvyLFy+qWrVqWrRo0V0/w7sJCgpSsWLFLOe9cePGLJ+D9L/7xb/99ltLWVhYmLp06aKff/5Z/fv3V+3atfWvf/1L0q1bIubOnav27durVq1aatq0qcLDw3X27NksMaxbt05t2rRRzZo11aNHDx05csRqe3a+T+nWFPB//vOfat26tWrWrKmQkBA9//zz+uWXX6zqHT58WEOGDFH9+vVVu3ZtDRgwQD/88EO2PzOz2ax//etfatKkierUqaORI0da9Ys5c+aoRo0aunr1apZ9X3vtNTVo0EApKSn3PU50dLSqVKmixo0bKyQkRNHR0Xesd/78eY0cOVJ16tRRSEiI3n77bX311VdZvq8HOfft27fr+PHjeuGFF7Ik8pmaNm1qdUEosy999913mjJlikJCQqy2r1q1Sp07d1bNmjXVtGlTvfHGG7p+/bpVm61bt1ZERESWY/315yuzf27duvWe3w0AOCJG5gHgEZCZCPn4+FjK0tPTLX+cT5gwQe7u7jIMQy+88IK+/fZb9ezZU9WqVdNXX32lGTNmKDY2VpMmTbLs/49//EObN29Wly5dVK9ePe3fv1/Dhw9/oDgjIyM1d+5c1a1bVy+++KJcXV11+PBh7d+/X02bNtWkSZP01ltvydPTUyNHjpSke64DsHv3bkmyGlG+l6+//loxMTHq3r27SpQood9//13r16/X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i4iIOImjR4/y008/sWvXLnbt2lVie0JCghHm75c9ZA8fPpzHHnvsnp/57wcAd6tbty4AZ8+epVWrVsb71atXp3r16gBUrFiRzMzMEt+9u1X+QdF69iIi8v8LhXkREREnkZCQgI+PDzNmzCixbd++fezbt49Zs2ZRtmxZ/P39OXbsGDk5OcVa5y9evFjse7Vq1QLAbDbTrl27/7lMoaGhrFq1ioSEhGJh/rfw9/cHIDk5mbZt2xbblpycbGyHOw8ILl26VGIf9tb9/5aSklLivQsXLlCjRo3fU2QREZFSozHzIiIiTiA3N5e9e/cSGhpK165dS/wMGDAAi8XCJ598AkCHDh0oKChgy5Ytxj5sNhvx8fHF9uvj40Pr1q15//33uX79eonfm5GR8YvlatWqFe3bt2fLli3s37//np8pKiq6r2Ns0qQJPj4+bN68mfz8fOP9gwcPkpSURGhoqPFerVq1OH/+fLHyfffdd5w4ceKe+96/f3+xsf+JiYl89dVXdOzY8b7KJiIi4mjUMi8iIuIEPvnkEywWS7E11O/WvHlzKleuzM6dO+nWrRthYWGEhIQwf/58Ll68SN26dfnkk0+4efMmcGfiPLtXX32V/v3706NHDyIiIqhVqxZpaWmcOnWKq1evsnPnzl8s28KFCxk5ciRRUVF07NiRdu3a4eXlRVpaGocPH+b48eP3FZrNZjOvvPIKU6ZMYeDAgXTv3t1Ymq5GjRoMHTrU+GyfPn1Yu3YtI0aMoE+fPqSnp7N582bq169fbBZ9u4CAACIjI4mMjCQ/P5/169dTqVIlRo4c+avlEhERcUQK8yIiIk5g586dlClThvbt299zu8lkIjQ0lISEBDIzM/H29uadd95h7ty5bN++HZPJROfOnYmKiioxW3z9+vXZunUry5YtY/v27dy4cYPKlSsTHBxMVFTUr5bN3pq+efNm9uzZw7Jly8jNzcXb25smTZqwaNEiunXrdl/HGR4eTtmyZVm9ejWLFi3Cw8ODsLAwJk6cWGxpu3r16jF//nzeeust5s2bR/369VmwYAEffvgh//73v0vst1evXphMJtatW0d6ejohISFER0dTrVq1+yqXiIiIo3Eput++byIiIuL09u/fT1RUFBs3bvzdY9ydQWpqKp06deL//u//GDFiRGkXR0RE5IHRmHkREZG/qNzc3GKvrVYrcXFxVKhQgcaNG5dSqURERORBUDd7ERGRv6g5c+aQm5tLixYtyM/PZ+/evZw8eZKXX375D1kWTkRERP48CvMiIiJ/UY8++ihr1qzhwIED5OXlUbt2baKjoxk4cGBpF01ERER+J42ZFxEREREREXEyGjMvIiIiIiIi4mQU5kVEREREREScjMK8iIiIiIiIiJNRmBcRERERERFxMgrzIiIiIiIiIk5GYV5ERERERETEySjMi4iIiIiIiDgZhXkRERERERERJ/P/APKvRncI0okoAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# This plot shows linearity of `Quantity` & `Total Amount`\n",
        "# Set Seaborn style\n",
        "sns.set(style=\"whitegrid\")\n",
        "\n",
        "# Create a scatterplot to visualize the correlation between Age and Quantity\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.scatterplot(data=df, x='Quantity', y='Total Amount', hue='Gender', alpha=0.7)\n",
        "plt.xlabel('Quantity')\n",
        "plt.ylabel('Total Amount')\n",
        "plt.title('Correlation between Quantity and Total Amount')\n",
        "plt.legend(title='Gender')\n",
        "\n",
        "plt.show()\n",
        "\n",
        "# Calculate the correlation coefficient between Age and Quantity\n",
        "correlation_coefficient = df['Quantity'].corr(df['Total Amount'])\n",
        "\n",
        "print(f\"Correlation coefficient between Quantity and Total Amount: {correlation_coefficient}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 590
        },
        "id": "N-O9HiBVx7mv",
        "outputId": "21b5fdbd-9e30-4c28-c678-234b38b39415"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ],
            "image/png": "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7hpXqomIiIiqGovlWqqTXxNMf6O91jzLDa3lGD2glUGmjXv33Xfx7rvvPrXP//73P/W/GzZsiG+++Uarz+DBg9X/dnZ2xuXLl7X69OvXD/369atEtEREREQVx2K5Fuvk1wTtfBzw7+XbeFggoKG1OVp42NW4K8pEREREtRWL5VrO2EiClu62MDMzg7GxsdjhEBEREdUpRmIHQERERERUU7FYJiIiIiLSgcUyEREREZEOLJaJiIiIiHRgsUxEREREpAOLZSIiIiIiHVgsExERERHpwGKZ9LZs2TJ4e3tr/QkLCxM7NDVvb29ERUWJHQYRERHVUnwoCVWKmZkZNmzYoNVGREREVBewWKZKMTIygr+/v9hhEBEREVULDsOo5QSVCkUp8Xh48Q/kXz8PQVUidkhqv/32GwYPHgw/Pz907NgRH3/8MfLy8tTLjx8/Dm9vb/z++++YNGkSAgIC0L17d0RHRwMANm7ciO7du6NDhw746KOPUFRUpF43PT0d06dPR8+ePeHn54c+ffrgq6++0uijb1xEREREpXhluRZ7eOkv3Nu/DiU599VtxpZ2aNhnJBo072iwOIqLizVeGxsbY9++fZgyZQpeeuklvPvuu7h79y6+/PJLZGdnY/HixRr958yZg0GDBuHVV1/F1q1b8eGHH+LSpUtISEjA3LlzkZKSggULFsDFxQXjxo0DADx48ADW1taYPn06FAoFkpOTsWzZMty9exfz58/XGevevXvLHRcRERERi+Va6uGlv3Bn++da7SU593Fn++dwfPkDgxTMeXl5aNmypUbbwoULsXTpUoSGhuKzzz5Tt9vb22PMmDF455138Nxzz6nbX3jhBUyYMAEA4OfnhwMHDmD37t04cOAApFIpAODvv//G3r171cWyt7c3pk2bpt5GmzZtIJfLERkZidmzZ0Mul2vFKggCFi1aVO64iIiIiGpUsXz9+nVERUXhzJkzSEhIgIeHB3bt2qVefvPmTfTs2bPMdWUyGc6dO/fUfq1bt8bWrVs12v755x8sXLgQ8fHxsLOzw9ChQzF69GhIJJIqPLKqJahKcG//uqf2uXdgHcy92kNiZFytsZiZmWHTpk0abSqVCqmpqZgxY4bGVecOHTrAyMgI58+f1yhKO3furP63paUlbG1t0a5dO3WhDABubm44fvy4+rUgCNiwYQO2bt2KmzdvorCw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eoR5FwfzLg7mvXr5+vrC0baBzuUNrcwhM5YY/P+X+ornuzgMmffy3ptW64rlJ5mbm6Nbt27Yt2+fuk2hUCAtLU2rb1ZWFqysrABAfeW59ApzqaKiIuTn56v76UMqlaJZs2Z6r18R+fn5SE5OhpubG+RyuUH2Scy7WJh3cTDvhlFcXIw2zR0QHZuAomKV1vLeHZuiiYMFHGx9RIiu/uD5Lg5D5z0xMbHcfWt9sVwWDw8PxMXFaV1iv3btGry8vAA8KrIbN26sNTb52rVrEARBayxzRUgkEoNP8SOXyzmtkAiYd3Ew7+Jg3qufU8MivP1KG6z931k8/P9jl40kQNc2rujk2xgymYwzNRkIz3dxGCrv5R2CAdSBYjkvLw+//fYbfH191W1du3bFihUrEBcXh06dOgF4VARfvHgRo0aN0uh36NAhfPDBB5BKpQCAmJgYKBQK9fhnIiIiQ7GxskYHHxM4OwTh9r085Bcp4dTQAg42pnBsqP83nkSkvxpVLOfn5+Po0aMAgNTUVOTm5mLv3r0AgA4dOiApKQlr165F79694eTkhPT0dKxfvx53797F119/rd5OQEAAgoODMWPGDEybNg2mpqZYvHgxvL290adPH3W/iIgIREdH4/3338fQoUNx5coVREVFYcqUKfzkTkREorCwsICXhQWc7eXIycmBpaUFr3ASiahGFcv379/HpEmTNNpKX2/cuBGNGjWCUqnE4sWLkZmZCblcjoCAAMydOxd+fn4a6y1ZsgTz58/H7NmzUVxcjODgYMycOVP99D4AaNq0KaKiorBgwQKMGTMGtra2mDhxIkaOHFn9B0tERPQMN2/ehI8PxygTialGFcvOzs64fPnyU/tERUWVa1uWlpaYN28e5s2b99R+bdq0wdatW8sdIxERERHVH7XyoSRERERERIbAYpmIiIiISAcWy0REREREOrBYJiIiIiLSgcUyEREREZEOLJaJiIiIiHRgsUxEREREpAOLZSIiIiIiHVgsExERERHpwGKZiIiIiEgHFstERERERDqwWCYiIiIi0oHFMhERERGRDiyWiYiIiIh0YLFMRERERKQDi2UiIiIiIh1YLBMRERER6cBimYiIiIhIBxbLREREREQ6sFgmIiIiItKBxTIRERERkQ4slomIiIiIdGCxTERERESkA4tlIiIiIiIdWCwTEREREenAYpmI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QRERERER1QR6Fcu//PILbty4oXP5zZs38euvv+obExERERFRjaBXsfws6enpMDMzq45NExEREREZTLnHLB88eBCHDh1Sv966dSv+/PNPrX45OTn4888/0apVq6qJkIiIiIhIJOUulq9evYq9e/cCACQSCc6cOYPz589r9JFIJDA3N0f79u05GwYRERER1XrlLpbHjh2LsWPHAgCaN2+Ozz77DP3796+2wIiIiIiIxKbX1HGXLl2q6jiIiIiIiGocvYrlUikpKYiNjcWtW7cAPJp/uWvXrnBxcamS4IiIiIiIxKR3sbxgwQJs3LgRKpVKo93IyAhvvPEGpk2bVungiIiIiIjEpFexvG7dOnz33Xd4/vnnMXLkSHh6egJ4dBPgd999h++++w6Ojo548803qzJWIiIiIiKD0qtY3rp1K0JCQvD1119rtLdu3RqLFy9GYWEhfvrpJxbLRERERFSr6fVQktTUVAQHB+tcHhwcjNTUVL2DIiIiIiKqCfQqlu3s7J46I8alS5dga2urd1BERERERDVBuYvlEydOICMjAwDwwgsv4Oeff8bq1auRl5en7pOXl4fVq1fj559/RmhoaNVHS0RERERkQOUeszxixAgsWrQI/fv3x6RJkxAfH4+vvvoKS5cuhYODAwAgPT0dxcXFCAwMxMSJE6staCIiIiIiQyh3sSwIgvrfcrkcGzZswMGDBzXmWQ4ODka3bt0QEhICiURS9dESERERERlQpR5K0qtXL/Tq1auqYiEiIiIiqlEqdIMfrxYTERERUX1SoSvLH3zwAT744INy9ZVIJLh48aJeQT1NeHg4/v777zKXffXVV+jXr5/OPjExMeoHqABATk4O5s+fj4MHD0KpVKJLly6YOXOmegw2EREREdVvFSqWO3XqBDc3t2oKpXw+/vhj5ObmarRt2LAB+/fvR1BQkLqtTZs2Wo/cdnZ21ng9efJkJCYmYs6cOTA1NcWSJUswevRobN++HSYmlRqhQkRERER1QIUqwoEDB6J///7VFUu5NGvWTKvt/fffR+fOnTXmdlYoFPD399e5ndOnT+OPP/5AVFSU+gEr7u7uCA0Nxf79+zn1HRERERHp91CSmuSff/7BzZs3K1zEx8bGQqFQoHPnzuo2Dw8P+Pj4IDY2tqrDJCIiIqJaqNYXy7t27YK5uTl69uyp0f7333/D398fvr6+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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Correlation coefficient between Quantity and Total Amount: 0.3737070541214061\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Select only the numeric columns for correlation\n",
        "numeric_columns = df.select_dtypes(include=['int64']).columns\n",
        "corr = df[numeric_columns].corr(method='spearman')\n",
        "# Create a mask to hide the upper triangle of the correlation matrix\n",
        "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
        "# Create a heatmap of the correlation matrix\n",
        "plt.figure(figsize=(10, 10))\n",
        "cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=\".2f\")\n",
        "cormat.set_title('Spearman Correlation Matrix')\n",
        "plt.show()\n",
        "('Correlation Matrix')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 857
        },
        "id": "Ws7e9Lagv5ih",
        "outputId": "be043dfa-3388-42df-e4ac-b0e6b55854a5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x1000 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAw4AAANICAYAAABuQ7g8AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACdzElEQVR4nOzdeVyUVfvH8e+AbIqACK5pKArilktqaqm5pqaZppnm3vq4lK3aqmluZWmSaaW55l7mkru/NDVzyS3DXdJKETdAdmF+f5jzyDPAzMAgw/B5v168wnPf576vmYPGNee6zzEYjUajAAAAACAbLvkdAAAAAADHR+IAAAAAwCISBwAAAAAWkTgAAAAAsIjEAQAAAIBFJA4AAAAALCJxAAAAAGARiQMAAAAAi0gcAAAAAFhE4gAABcivv/6q0NBQ/frrr3a9bmhoqKZNm2bXaxZkI0aMUMuWLfM7DABwKCQOgIM6fvy4hg0bpocffli1atXSQw89pAEDBmj+/Pn5HZrTuXz5siZOnKhHHnlE9913n+rUqaOuXbtq+vTpio2Nze/w7Gbbtm0OlxxMmzZNoaGhqlatmi5cuGB2/MaNG6pdu7ZCQ0P1wQcf2Hz9xMRETZs2ze6JFgAURkXyOwAA5n777Tf17dtX5cqVU/fu3RUYGKgLFy7o0KFDmjdvnvr06ZPfITqNw4cP67nnnlNCQoI6d+6sGjVqSJJ+//13ffXVV9q3b59mz56dz1Hax7Zt27Rw4UINHTrU7Njhw4fl6uqaD1Hd4u7urjVr1ujZZ5/N0L5x48ZcXTcxMVHh4eEaMmSIGjVqZHW/MWPGyGg05ureAOBsSBwABzRjxgwVL15cy5cvl4+PT4ZjV65cyaeoLLt586bS09Pl7u6e36FYJTY2VkOGDJGrq6u+//57BQcHZzg+fPhwLV261C73SkxMlJeXl1m7o7xnHh4e+Xr/5s2ba+3atWaJw5o1a9SiRQtt2LDhrsSRkJCgokWLys3N7a7cDwAKEkqVAAd07tw5ValSxSxpkKSSJUtm+PPtEo5Vq1apXbt2qlWrlrp27aq9e/ea9Y2KitLIkSPVpEkT1axZUx07dtTy5csznJOSkqKpU6eqa9euql+/vurUqaNevXpp9+7dGc7766+/FBoaqlmzZmnOnDlq3bq1atWqpdOnT5vKT86ePavXXntN9evX1wMPPKApU6bIaDTqwoULevHFF1WvXj01bdrU7BP9nMSwZMkStW7dWjVr1lS3bt10+PBhi+/z4sWLFRUVpREjRpglDZIUEBCg//znPxnaFi5cqI4dO6pmzZp68MEHNXr0aLNypj59+ujRRx/V77//rt69e+u+++7TJ598ku17JkmnT5/WsGHD1LBhQ9M4btmyxeLr2Ldvn4YNG6YWLVqoZs2aat68ucaNG6ekpCTTOSNGjNDChQsl3fqZuf11W2bPOPzxxx965plnVK9ePdWtW1f9+vXTwYMHM5zz3XffKTQ0VPv379f48eP1wAMPqE6dOho8eLCuXr1qMfbbHn30UUVERJjeC0mKjo7W7t279eijj5qdb83PyF9//aXGjRtLksLDw02v+fbrHDFihOrWratz587p2WefVd26dfXaa6+Zjt35jMNnn32matWq6ZdffskQx7vvvquaNWvq2LFjVr9WACiomHEAHFD58uV14MABnThxQiEhIRbP37t3r3788Uf16dNH7u7uWrRokZ555hktW7bM1P/y5cvq0aOHDAaDevfuLX9/f23fvl1vv/22bty4of79+0u6VVO+bNkyPfroo+revbvi4+O1fPly0/XCwsIy3Pu7775TcnKyevToIXd3d/n6+pqODR8+XMHBwXr11Ve1bds2ffHFF/Lz89PixYv1wAMP6LXXXtPq1as1ceJE1apVSw0aNMhRDGvWrFF8fLyefPJJGQwGff311xo6dKg2b96c7SfHW7dulaenp9q1a2fVuEybNk3h4eFq0qSJnnrqKZ09e1aLFi3SkSNHtGjRogz3un79up599ll17NhRnTt3zpDwZfaenTx5Uk899ZRKly6tZ599VkWLFtW6des0ePBgTZs2TW3atMkyrvXr1yspKUlPPfWU/Pz8dPjwYS1YsEAXL17UZ599Jkl68skndenSJe3cuVOTJk2y+FpPnjyp3r17q1ixYnrmmWdUpEgRLVmyRH369NGCBQt03333ZTh/7Nix8vHx0ZAhQ/T3339r7ty5+uCDDzRlyhSr3tsGDRqoTJkyWrNmjV566SVJ0o8//qiiRYuqRYsWZudb8zPi7++vUaNGadSoUWrTpo3pPbwzYbp586YGDRqk+vXr680335Snp2em8b344ov6v//7P7399ttatWqVvL299fPPP2vp0qV66aWXVK1aNateJwAUaEYADmfHjh3GsLAwY1hYmPHJJ580Tpo0yfjzzz8bU1JSzM4NCQkxhoSEGI8cOWJq+/vvv421atUyDh482NT21ltvGZs2bWq8evVqhv7Dhw831q9f35iYmGg0Go3GmzdvGpOTkzOcExMTY2zSpIlx5MiRprbz588bQ0JCjPXq1TNeuXIlw/mfffaZMSQkxPjuu++a2m7evGls1qyZMTQ01Dhz5swM165du7bxzTffzHCuLTE0bNjQeP36dVP75s2bjSEhIcatW7eavV93atCggbFz587ZnnPblStXjDVq1DAOHDjQmJaWZmpfsGCBMSQkxLh8+XJT29NPP20MCQkxLlq0KMM1snvP+vXrZ3z00UczvO709HTjk08+aWzbtq2pbffu3caQkBDj7t27TW23x+5OM2fONIaGhhr//vtvU9vo0aONISEhmb6+kJAQ42effWb683/+8x9jjRo1jOfOnTO1RUVFGevWrWvs3bu3qW3FihXGkJAQY//+/Y3p6emm9nHjxhnDwsKMsbGxmd7vtts/K1euXDFOmDDB2KZNG9Oxbt26GUeMGGGKb/To0aZj1v6MXLlyxey13fbmm28aQ0JCjB9//HGmxx5++OEMbcePHzfWqFHD+PbbbxtjYmKMDz30kLFr167G1NTUbF8jADgLSpUAB9S0aVMtXrxYLVu21LFjx/T1119r0KBBatasWaalK3Xr1lXNmjVNfy5XrpxatWqlHTt2KC0tTUajURs3blTLli1lNBp19epV09eDDz6ouLg4HT16VJLk6upqqrdPT0/X9evXdfPmTdWsWVN//PGH2b3btm0rf3//TF/HE088Yfre1dVVNWvWlNFozNDu4+OjSpUq6fz58xnOtSWGDh06ZJjpuP/++yUpwzUzc+PGDRUrVizbc27btWuXUlNT1bdvX7m4/Pefzu7du8vb21vbtm3LcL67u7u6du2a6bX+9z27fv26du/erfbt2+vGjRumsbl27ZoefPBBRUZGKioqKsvY7vyUPCEhQVevXlXdunVlNBozfb8sSUtL086dO9W6dWtVqFDB1F6qVCk9+uij2r9/v27cuJGhz+3ZrNvuv/9+paWl6e+//7b6vp06ddKff/6pw4cP688//9SRI0fUqVOnTM+19WckO0899ZRV54WEhGjYsGFatmyZBg0apGvXrmnixIkqUoTJewCFA//aAQ6qdu3aCg8PV0pKio4dO6bNmzdrzpw5eumll7Ry5UpVqVLFdO69995r1j8oKEiJiYm6evWqXFxcFBsbqyVLlmjJkiWZ3u/OevTvv/9es2fP1tmzZ5Wammpqv+eee8z6ZdZ2W7ly5TL8uXjx4vLw8DBLNIoXL67r169naLMlhrJly2b48+0kwtJSqt7e3oqPj8/2nNv++ecfSVLlypUztLu7u6tChQpmvyCXLl06ywee//c1nDt3TkajUVOnTtXUqVMz7XPlyhWVLl06y9g+++wzbd26VTExMRmO/e8v+Na4evWqEhMTValSJbNjwcHBSk9P14ULF1S1alVT+/+O9e3nc2xZzrZ69eqqXLmy1qxZIx8fHwUGBuqBBx7I8nxbfkayUqRIEZUpU8bq8wcNGqS1a9fq8OHDeuWVVzL8PQQAZ0fiADg4d3d31a5dW7Vr11ZQUJBGjhyp9evXa8iQIVZfIz09XZLUuXNnPf7445mec7vu+4cfftCIESPUunVrDRo0SCVLlpSrq6tmzpyZ6Sf4WdWES8rwyfxtWS35abxj6UtbY7DmmpmpXLmyIiIilJKSYvdVjbJ7X/732O3xGThwoB566KFM+1SsWDHT9rS0NA0YMEAxMTF65plnVLlyZRUtWtT00Pfta+e1zMZasjwG/+vRRx/VokWLVKxYMbVv3z7L69r6M5IVd3f3LO+RmfPnz+vPP/+UJJ04ccLqfgDgDEgcgALkdjnSpUuXMrTf/kXmTpGRkfLy8jJ9ul+sWDGlp6erSZMm2d5jw4YNqlChgsLDwzOUntx+yPZuuFsxPPzwwzpw4IA2btyY6co9d7r9ifqZM2cylO+kpKTor7/+svi+Zuf29dzc3Gy+zokTJxQZGamJEyeqS5cupvadO3eanXvne5kdf39/eXl56ezZs2bHzpw5IxcXF7NZHnvp1KmTPvvsM0VHR+ujjz7K8jxrf0asfc3WSE9P14gRI+Tt7a1+/fppxowZateundq2bWu3ewCAI+MZB8AB7d69O9NPam/X0f9vucyBAwdMzyhI0oULF7RlyxY1bdpUrq6ucnV1Vbt27bRhw4ZMPyW9s0zp9qf3d97/0KFDZstw5qW7FUPPnj0VGBioCRMmZPpL8pUrVzR9+nRJUpMmTeTm5qb58+dniGv58uWKi4tT8+bNcxxHyZIl1bBhQy1ZssQsKZSU7bKmtz8tvzMmo9GoefPmmZ17ex8JS+VDrq6uatq0qbZs2aK//vrL1H758mWtWbNG9evXl7e3d/YvKocqVqyot956S6+++qpq166dbYyS5Z8Ra1+zNb755hsdOHBAH3zwgV566SXVrVtXo0aNsmnZWQAoyJhxABzQ2LFjlZiYqDZt2qhy5cpKTU3Vb7/9pnXr1ql8+fJmD92GhIRo0KBBGZZjlZRhh+BXX31Vv/76q3r06KHu3burSpUqiomJ0dGjR/XLL79oz549kqQWLVpo48aNGjx4sFq0aKG//vpLixcvVpUqVZSQkHBXXv/disHX11eff/65nnvuOXXp0iXDztF//PGH1qxZo7p160q69Sn8888/r/DwcD3zzDNq2bKlzp49q2+//Va1atVS586dcxXL+++/r169eqlTp07q0aOHKlSooMuXL+vgwYO6ePGiVq1alWm/ypUrq2LFipo4caKioqLk7e2tDRs2ZPqL8u3XNnbsWD344INydXVVx44dM73uyy+/rF27dqlXr17q1auXXF1dtWTJEqWkpOj111/P1Wu1pF+/fhbPsfZnxNPTU1WqVNG6desUFBQkPz8/Va1a1aplju90+vRp074Rt/d3mDBhgrp06aLRo0dn+WwKADgTEgfAAb3xxhtav369tm3bpiVLlig1NVXlypVTr1699OKLL5ptDNegQQPVqVNHn3/+uf755x9VqVJF48ePz7C2fEBAgJYtW6bPP/9cmzZt0qJFi+Tn56cqVaqYNr2SpK5du+ry5ctasmSJduzYoSpVquijjz7S+vXrTclFXrubMdx3331avXq1Zs2apZ9++kk//PCDXFxcVLlyZT333HN6+umnTecOHTpU/v7+WrBggcaPHy9fX1/16NFDr7zySq53Gq5SpYpWrFih8PBwff/997p+/br8/f1VvXp1DR48OMt+bm5umjFjhsaOHauZM2fKw8NDbdq0Ue/evfXYY49lOLdt27bq06eP1q5dq1WrVsloNGaZOFStWlULFy7U5MmTNXPmTBmNRtWuXVsfffSR2R4O+cGWn5GxY8dqzJgxGj9+vFJTUzVkyBCbEoe0tDS9+eabKlGihN566y1Te1BQkF555RV9+OGH+vHHH9WhQwe7vT4AcEQGo61PrgFwKKGhoerdu7fee++9/A4FAAA4MZ5xAAAAAGARiQMAAAAAi0gc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EhERERFR91EpcXj22Wfx7LPPArjX4zBnzhy5x1oTEREREfUkovbfyktKdOg5DgDQ0tKCS5cuobq6GkZGRrC2tuawV0REREREfVSHbo7+4osvsHnzZlRWVqKlpQUikQimpqaIiIjgPQ5ERERERH1QuxOH9PR0JCQk4IUXXsCMGTMwaNAg3LhxA3l5eUhISIC2tjbmzJnTFbESEREREamIlTDq1u7EYdeuXQgMDMSKFSvkpnt7e8PExAQpKSlMHIiIiIiI+ph23zVSWlr6wJGTpkyZgqtXr3Y6KCIiIiIi6lnanTiYmZmhqKiozffOnj0LMzOzTgdFRERERNQZIpGoR/7rzVQqVcrOzoanpyeMjY3h5+eHzZs3o76+Hs899xxMTU1RWVmJI0eOICUlBZGRkV0dMxERERERPWIqJQ5xcXFIT0+HsbExwsPDUVNTg5SUFGzbtk2YR1NTE4GBgQgPD++yYImIiIiIqHuolDi0tLQI/xeJRHjnnXcQFhaG4uJi4TkOjo6OMDY27rJAiYiIiIhU17vLgnqiDj3HAQCMjY3h6empzliIiIiIiKiHUjlxkEgk+Omnn5TOJxKJEBIS0pmYiIiIiIioh1E5cdi9e7dK8zFxICIiIqLuJmr/4KGkhMqJQ0ZGBhwdHbsyFiIiIiIi6qGYihERERERkVIdvjmaiIiIiKjn4qhK6sYeByIiIiIiUkqlHofz5893dRxERERERNSDsVSJiIiIiPocEUuV1I6lSkREREREpBQTByIiIiIiUoqlSkRERETU54hELFVSN/Y4EBERERGRUkwciIiIiIhIKZYqEREREVEfxOvj6sZvlIiIiIiIlGLiQERERERESrFUiYiIiIj6HD4ATv3Y40BEREREREoxcSAiIiIiIqVYqkREREREfRBLldSNPQ5ERERERKQUEwciIiIiIlKKpUpERERE1OeIRCxVUjf2OBARERERkVJMHIiIiIiISCmWKhERERFRH8Tr4+rGb5SIiIiIiJRi4kBEREREREqxVImIiIiI+hwRHwCnduxxICIiIiIipUQtLS0t3R0EEREREZF6/d7dATzAiO4OoMOYOBARERERkVIsVSIiIiIiIqWYOBARERERkVJMHIiIiIiISCkmDkREREREpBQTByIiIiIiUoqJAxERERERKcXEgYiIiIiIlGLiQERERERESjFxICIi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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**CRAMER'S RULE**"
      ],
      "metadata": {
        "id": "UQ6W2Q7_y4Vk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from scipy.stats import chi2_contingency\n",
        "from math import sqrt\n",
        "\n",
        "# Create a contingency table\n",
        "contingency_table = pd.crosstab(df['Gender'], df['Product Category'])\n",
        "\n",
        "# Perform the chi-square test\n",
        "chi2, p, _, _ = chi2_contingency(contingency_table)\n",
        "\n",
        "# Calculate Cramer's V\n",
        "num_obs = np.sum(contingency_table.values)\n",
        "cramer_v = sqrt(chi2 / (num_obs * (min(contingency_table.shape) - 1)))\n",
        "\n",
        "print(f\"Chi-Square Value: {chi2}\")\n",
        "print(f\"P-Value: {p}\")\n",
        "print(f\"Cramer's V: {cramer_v}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fipCE5ER-wIJ",
        "outputId": "02885ed4-8417-4edc-b230-fc0a047f7ade"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Chi-Square Value: 1.673837085800602\n",
            "P-Value: 0.43304287262068974\n",
            "Cramer's V: 0.04091255413440478\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "r_p309nY4JqZ",
        "outputId": "e43d2b8b-3c71-4aab-c472-cb4fd44b22bc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     Age  Price per Unit  Gender_Male  Product_Clothing  Product_Electronics\n",
              "29    39             300            0                 0                    0\n",
              "535   55              30            0                 0                    0\n",
              "695   50              50            0                 1                    0\n",
              "557   41              25            0                 1                    0\n",
              "836   18              30            1                 0                    0\n",
              "..   ...             ...          ...               ...                  ...\n",
              "106   21             300            0                 1                    0\n",
              "270   62              30            0                 0                    0\n",
              "860   41              30            0                 1                    0\n",
              "435   57              30            0                 1                    0\n",
              "102   59              25            0                 1                    0\n",
              "\n",
              "[800 rows x 5 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e9d999c7-e04f-404b-9287-aae13e92f73c\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Price per Unit</th>\n",
              "      <th>Gender_Male</th>\n",
              "      <th>Product_Clothing</th>\n",
              "      <th>Product_Electronics</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>29</th>\n",
              "      <td>39</td>\n",
              "      <td>300</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>535</th>\n",
              "      <td>55</td>\n",
              "      <td>30</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>695</th>\n",
              "      <td>50</td>\n",
              "      <td>50</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>557</th>\n",
              "      <td>41</td>\n",
              "      <td>25</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>836</th>\n",
              "      <td>18</td>\n",
              "      <td>30</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>106</th>\n",
              "      <td>21</td>\n",
              "      <td>300</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>270</th>\n",
              "      <td>62</td>\n",
              "      <td>30</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>860</th>\n",
              "      <td>41</td>\n",
              "      <td>30</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>435</th>\n",
              "      <td>57</td>\n",
              "      <td>30</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>102</th>\n",
              "      <td>59</td>\n",
              "      <td>25</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>800 rows × 5 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e9d999c7-e04f-404b-9287-aae13e92f73c')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e9d999c7-e04f-404b-9287-aae13e92f73c button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e9d999c7-e04f-404b-9287-aae13e92f73c');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-c9bdecb8-396b-47ba-b798-c5cfb148c9a9\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-c9bdecb8-396b-47ba-b798-c5cfb148c9a9')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-c9bdecb8-396b-47ba-b798-c5cfb148c9a9 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 91
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**MACHINE LEARNING MODEL EVALUATION**"
      ],
      "metadata": {
        "id": "5Jwei_ZDy-J5"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**DECISION TREE REGRESSOR MODEL**"
      ],
      "metadata": {
        "id": "9uJjfsauL9nt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeRegressor\n",
        "from sklearn.metrics import mean_squared_error,mean_absolute_error\n",
        "\n",
        "# Building a decision tree model\n",
        "model = DecisionTreeRegressor(random_state=42)\n",
        "model.fit(X_train, y_train_numerical)\n",
        "\n",
        "# Making predictions\n",
        "y_preddt = model.predict(X_test)\n",
        "\n",
        "# Evaluating the model\n",
        "mse = mean_squared_error(y_test_numerical, y_preddt)\n",
        "mae=mean_absolute_error(y_test_numerical,y_preddt)\n",
        "print(\"Mean Squared Error:\", mse)\n",
        "print(\"Mean Absolute Error:\",mae)\n",
        "dtr=DecisionTreeRegressor()\n",
        "dtr.fit(X_train,y_train_numerical)\n",
        "scoredt=dtr.score(X_train,y_train_numerical)\n",
        "print(\"Accuracy of  DecisionTreeRegressor:\",scoredt)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "psRVWEefFOsh",
        "outputId": "a4638840-4c7c-4455-e7cd-7c473ecbb632"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error: 2.7230555555555553\n",
            "Mean Absolute Error: 1.33\n",
            "Accuracy of  DecisionTreeRegressor: 0.7627112551205317\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**LINEAR REGRESSION MODEL**"
      ],
      "metadata": {
        "id": "yPgZbqN-O-PI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "# Assuming 'Quantity' is the target variable\n",
        "y_train_numerical = df.loc[X_train.index, 'Quantity']\n",
        "y_test_numerical = df.loc[X_test.index, 'Quantity']\n",
        "\n",
        "lr = LinearRegression()\n",
        "lr.fit(X_train, y_train_numerical)\n",
        "scorelr=lr.score(X_train,y_train_numerical)\n",
        "\n",
        "# Making predictions\n",
        "y_pred = lr.predict(X_test)\n",
        "\n",
        "# Evaluating the model\n",
        "mse = mean_squared_error(y_test_numerical, y_pred)\n",
        "print(\"Mean Squared Error:\", mse)\n",
        "mae=mean_absolute_error(y_test_numerical,y_pred)\n",
        "print(\"Mean Absolute Error:\",mae)\n",
        "print(\"Accuracy of LinearRegressor:\",scorelr)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Jrv87Es43DqL",
        "outputId": "9f201215-4ace-4c9d-cdda-80209f7be861"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error: 1.3189859739999195\n",
            "Mean Absolute Error: 1.0310419160250914\n",
            "Accuracy of LinearRegressor: 0.002567275141953207\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**RANDOM FOREST CLASSIFIER MODEL**"
      ],
      "metadata": {
        "id": "N0GG0DTYPHCz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import mean_squared_error,mean_absolute_error\n",
        "\n",
        "# Building a decision tree model\n",
        "model = RandomForestClassifier(random_state=42)\n",
        "model.fit(X_train, y_train_numerical)\n",
        "\n",
        "# Making predictions\n",
        "y_preddt = model.predict(X_test)\n",
        "\n",
        "# Evaluating the model\n",
        "mse = mean_squared_error(y_test_numerical, y_preddt)\n",
        "mae=mean_absolute_error(y_test_numerical,y_preddt)\n",
        "print(\"Mean Squared Error:\", mse)\n",
        "print(\"Mean Absolute Error:\",mae)\n",
        "rfc=RandomForestClassifier()\n",
        "rfc.fit(X_train,y_train_numerical)\n",
        "scorerfc=rfc.score(X_train,y_train_numerical)\n",
        "print(\"Accuracy of  RandomForestClassifier:\",scorerfc)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xYanMdRCNp7P",
        "outputId": "918a17d8-0453-4ceb-8bb9-a6ae97de3aff"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error: 2.66\n",
            "Mean Absolute Error: 1.3\n",
            "Accuracy of  RandomForestClassifier: 0.83375\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SUPPORT VECTOR MACHINE**"
      ],
      "metadata": {
        "id": "lOOxp69_5sIu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVR\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "\n",
        "# Assuming 'Quantity' is the target variable\n",
        "y_train_numerical = df.loc[X_train.index, 'Quantity']\n",
        "y_test_numerical = df.loc[X_test.index, 'Quantity']\n",
        "\n",
        "svr = SVR(kernel='rbf')\n",
        "svr.fit(X_train, y_train_numerical)\n",
        "\n",
        "# Making predictions\n",
        "y_preds = svr.predict(X_test)\n",
        "scores=svr.score(X_train,y_train_numerical)\n",
        "\n",
        "# Evaluating the model\n",
        "mse = mean_squared_error(y_test_numerical, y_preds)\n",
        "mae = mean_absolute_error(y_test_numerical, y_preds)\n",
        "\n",
        "print(\"Mean Squared Error:\", mse)\n",
        "print(\"Mean Absolute Error:\", mae)\n",
        "print(\"Accuracy of svm: \",scores)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xkopjNjk1w8P",
        "outputId": "2cd57b5d-a1ec-4285-f6af-00d708d09b40"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean Squared Error: 1.5213381644188433\n",
            "Mean Absolute Error: 1.0425069234749194\n",
            "Accuracy of svm: 0.7295313917\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Accuracy of  DecisionTreeRegressor:\",scoredt)\n",
        "print(\"Accuracy of LinearRegressor:\",scorelr)\n",
        "print(\"Accuracy of  RandomForestClassifier:\",scorerfc)\n",
        "print(\"Accuracy of Support Vector Machine:\",scores)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nrfY2Y2EAbXC",
        "outputId": "6a6b1e0f-6e6f-4419-9100-9b3a40168ffa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of  DecisionTreeRegressor: 0.83375\n",
            "Accuracy of LinearRegressor: 0.002567275141953207\n",
            "Accuracy of  RandomForestClassifier: 0.83375\n",
            "Accuracy of Support Vector Machine:0.7295313917\n"
          ]
        }
      ]
    }
  ]
}