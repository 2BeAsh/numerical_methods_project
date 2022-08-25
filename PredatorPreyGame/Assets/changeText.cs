using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class changeText : MonoBehaviour
{

    private TMP_Text textContent;
    public string TextOnHover;
    private Image ImageObj;


    // Start is called before the first frame update
    void Start()
    {
        textContent = GetComponentInChildren<TMP_Text>();
        ImageObj = GetComponent<Image>();

    }


    private void OnMouseOver()
    {
        textContent.text = TextOnHover;
        ImageObj.color = Color.red;
    }


}
