using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Prey : MonoBehaviour
{
    Animator animator;
    public AudioSource audioPlayerCollide;
    public AudioSource audioPlayerDeath;
    public ParticleSystem bloodParticleSys;
    
    public bool once = true;

    List<AudioSource> AudioObjectsDeath = new();

    public float Health
    {
        set
        {
            health = value;
            if (health <= 0)
            {
                Defeated();
            }
        }
        get
        {
            return health;
        }

    }

    public float health = 1;
    public int scoreIncreaseAmount;

    public void Start()
    {
        animator = GetComponent<Animator>();

        // Get all objects that contain audio files and append their AudioSource component to a list
        foreach (GameObject fooObj in GameObject.FindGameObjectsWithTag("PreyDeathAudio")) 
        {
            AudioSource AudioObject = fooObj.GetComponent<AudioSource>();

            AudioObjectsDeath.Add(AudioObject);
        }
    }

    private void playDeathSound()
    {
        int AudioOptions = AudioObjectsDeath.Count;
        int PlayIndex = Random.Range(0, AudioOptions);
        AudioSource PlayAudio = AudioObjectsDeath[PlayIndex];
        PlayAudio.Play();
    }

    public void Defeated()
    {
        bloodParticleEffect();
        scoreCounter.instance.increaseScore(scoreIncreaseAmount);
        playDeathSound();
        //audioPlayerDeath.Play();
         
        animator.SetTrigger("Defeated");

    }

    public void RemoveEnemy()
    {
        Destroy(gameObject);
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        if(collision.gameObject.tag == "Player")
        {
            audioPlayerCollide.Play();
        }
    }


    public void bloodParticleEffect()
    {
        var em = bloodParticleSys.emission;
        em.enabled = true;
        bloodParticleSys.Play();
        once = false;
    }


}
