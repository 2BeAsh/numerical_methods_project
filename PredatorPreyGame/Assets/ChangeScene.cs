using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ChangeScene : MonoBehaviour
{

    public void ExitButton()
    {
        Debug.Log("Game closed");
        Application.Quit();
            
    }

    public void StartGame()
    {
        SceneManager.LoadScene("Game");
    }

    public void CharacterSelection()
    {
        SceneManager.LoadScene("CharacterInfo");
    }

    public void Credits()
    {
        SceneManager.LoadScene("Credits");
    }


    public void MainMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }


}
