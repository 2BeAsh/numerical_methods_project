using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movePeanut : MonoBehaviour
{

    Rigidbody2D rb;

    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Vector2 velocity = new Vector2(1, 0);
        rb.MovePosition(rb.position + velocity * Time.fixedDeltaTime);
    }
}
